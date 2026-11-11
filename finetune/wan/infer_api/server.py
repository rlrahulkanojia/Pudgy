#!/usr/bin/env python3
"""Pudgy Penguins inference API: generate videos with the trained model on a GPU box.

One GPU, one job at a time: jobs wait in a queue, the model stays loaded between jobs, and each job
runs its takes back to back (a longer video = takes chained on the previous take's last frame, as
the client long-form videos were made). Every finished video is checked (frame count), saved to
the dashboard's Azure container (so it outlives the box) and listed for the dashboard.

The dashboard finds the server through Azure: while running, the server writes its public address
and a heartbeat to <dashboard container>/inference/server.json every 30 s, and marks it stopped on
a clean exit. A box that is switched off stops heart-beating, which the dashboard reports as "no
server available". The public address is a Cloudflare quick tunnel (--tunnel; HTTPS, changes on
every start, hence the registration) or any address given with --public-url.

    # on the GPU box (finetune/wan/infer_api/serve.sh wraps this, in tmux, with the right venv):
    python server.py --backend wan --tunnel --register --upload
    # on a laptop, no GPU, for trying the dashboard:
    python server.py --backend mock --no-auth --port 8765

Auth: every endpoint except /v1/health and the docs needs the key in INFERENCE_API_KEY (environment
or the repo .env), sent as "X-API-Key: <key>" or "Authorization: Bearer <key>". Interactive docs at /docs.
Guide: finetune/wan/infer_api/README.md.
"""
import argparse
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import uvicorn  # noqa: E402
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import backends as B  # noqa: E402
import prompts as PR  # noqa: E402

VERSION = "1.0"
FPS = 24
SIZES = {   # key -> (width, height, label); 720x1280 is the client reels format (long-form videos)
    "portrait_720": (720, 1280, "Portrait 720×1280 (reels)"),
    "portrait_480": (480, 848, "Portrait 480×848 (fast preview)"),
    "landscape_720": (1280, 720, "Landscape 1280×720"),
    "landscape_480": (848, 480, "Landscape 848×480 (fast preview)"),
}
FRAMES = (33, 49, 81)        # 4k+1; 81 = one take of 3.4 s, the longest that fits (241 in one go OOMs)
MAX_TAKES = 3                # 3 x 81 - 2 seam frames = 241 frames = 10 s, as the long-form videos
MODES = {"chain": "continuous: each take starts on the previous take's last frame",
         "loop": "seamless loop: every take starts and ends on the start frame",
         "first_last": "first + last frame: one take from the start frame to your end frame"}
MAX_QUEUE = 10
MAX_IMAGE_BYTES = 20 * 1024 * 1024
ASPECT_TOLERANCE = 1.15      # a start frame whose shape differs more than this from the output is refused
REGISTRY_BLOB = "inference/server.json"
HEARTBEAT_S = 30


def utc(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t if t is not None else time.time()))


def log(*a):
    print(f"[{utc()}]", *a, flush=True)


def read_env_file(path):
    """NAME=value lines only - the repo .env also holds notes and code, so it is never executed."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(errors="replace").splitlines():
        m = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if m and m.group(1) not in os.environ:
            v = m.group(2).strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            os.environ[m.group(1)] = v


def ffprobe_frames(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", str(p)], capture_output=True, text=True)
    try:
        return int(r.stdout.strip().split(",")[0])
    except ValueError:
        return -1


def ffmpeg(*args):
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        raise B.GenError(f"ffmpeg failed: {r.stderr.strip()[-400:]}")


def last_frame(mp4, png):
    ffmpeg("-sseof", "-1", "-i", mp4, "-update", "1", "-q:v", "1", png)
    return png


def join_takes(takes, out):
    """Concatenate takes; takes 2..n start on the previous take's last frame (chain) or on the start
    frame the previous take ended on (loop), so their first frame is dropped. H.264, web-streamable."""
    ins, parts = [], []
    for i, t in enumerate(takes):
        ins += ["-i", t]
        trim = "trim=start_frame=1," if i else ""
        parts.append(f"[{i}:v]{trim}setpts=PTS-STARTPTS[v{i}]")
    graph = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(takes))) + \
        f"concat=n={len(takes)}:v=1:a=0[out]"
    ffmpeg(*ins, "-filter_complex", graph, "-map", "[out]", "-r", FPS, "-c:v", "libx264", "-crf", "16",
           "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out)


# ---------------------------------------------------------------------------------------------
class Azure:
    """The dashboard's container: registration + finished videos. Off when no connection string."""

    def __init__(self):
        self.cc, self.error = None, None
        conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        self.container = os.environ.get("AZURE_DASHBOARD_CONTAINER", "pudgy-dashboard")
        if not conn:
            self.error = "AZURE_STORAGE_CONNECTION_STRING is not set"
            return
        try:
            from azure.storage.blob import BlobServiceClient
            self.cc = BlobServiceClient.from_connection_string(conn).get_container_client(self.container)
        except Exception as e:
            self.error = f"Azure client failed: {e}"

    def put(self, name, data, content_type):
        from azure.storage.blob import ContentSettings
        bc = self.cc.get_blob_client(name)
        bc.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type=content_type))
        return bc

    def put_file(self, name, path, content_type):
        with open(path, "rb") as f:
            bc = self.put(name, f, content_type)
        size = bc.get_blob_properties().size
        if size != Path(path).stat().st_size:
            raise RuntimeError(f"{name}: uploaded {size} bytes, file has {Path(path).stat().st_size}")


class Jobs:
    """Job records in memory + <work>/jobs/<id>/job.json (survives a server restart)."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Condition()
        self.jobs, self.queue = {}, deque()
        for f in sorted(self.root.glob("*/job.json")):
            try:
                j = json.loads(f.read_text())
            except Exception:
                continue
            if j["state"] == "queued":
                self.queue.append(j["id"])
            elif j["state"] == "running":
                j.update(state="interrupted", stage="interrupted", finished_utc=utc(),
                         error="The server restarted while this job was running; submit it again.")
            self.jobs[j["id"]] = j
            self._save(j)

    def dir(self, jid):
        return self.root / jid

    def _save(self, j):
        p = self.dir(j["id"]) / "job.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(j, indent=1))
        tmp.replace(p)

    def add(self, j):
        with self.lock:
            self.jobs[j["id"]] = j
            self.queue.append(j["id"])
            self._save(j)
            self.lock.notify_all()

    def update(self, jid, **kw):
        with self.lock:
            self.jobs[jid].update(kw)
            self._save(self.jobs[jid])
            return dict(self.jobs[jid])

    def next(self, timeout=1.0):
        with self.lock:
            if not self.queue:
                self.lock.wait(timeout)
            while self.queue:
                jid = self.queue.popleft()
                if self.jobs[jid]["state"] == "queued":
                    return jid
            return None

    def get(self, jid):
        with self.lock:
            return dict(self.jobs[jid]) if jid in self.jobs else None

    def position(self, jid):
        with self.lock:
            return list(self.queue).index(jid) + 1 if jid in self.queue else 0

    def recent(self, n):
        with self.lock:
            return [dict(self.jobs[k]) for k in sorted(self.jobs, reverse=True)[:n]]

    def queued_ids(self):
        with self.lock:
            return list(self.queue)

    def remove_from_queue(self, jid):
        with self.lock:
            if jid in self.queue:
                self.queue.remove(jid)


class Service:
    def __init__(self, a):
        self.a = a
        self.work = Path(a.work)
        self.started = utc()
        self.jobs = Jobs(self.work / "jobs")
        self.keyframes_dir = Path(a.keyframes) if a.keyframes else self.work / "keyframes"
        if a.backend == "mock":
            self.backend = B.MockBackend(self.work, step_seconds=a.mock_step_seconds)
            self.base_take_s, self.load_s = 25 * a.mock_step_seconds + 1.0, 1.5
        else:
            self.backend = B.WanBackend(a.ws, self.work, gpu=a.gpu, models=a.models.split(","))
            # 720x1280, 81 frames, 25 steps on an A100 80GB (client long-form batch, Oct 2026), and a
            # model load; both are re-fitted from this server's own takes (calibration.json).
            self.base_take_s, self.load_s = 1800.0, 300.0
        self.cal_path = self.work / "calibration.json"
        self.cal = json.loads(self.cal_path.read_text()).get(self.backend.name, 1.0) if self.cal_path.exists() else 1.0
        self.public_url = a.public_url.rstrip("/") if a.public_url else None
        self.azure = Azure() if (a.register or a.upload) else None
        self.cancels = {}
        self.running = None
        self.stop = threading.Event()
        self._models = None
        self.tunnel_proc = None

    # ---- estimates ------------------------------------------------------------------------------
    @staticmethod
    def tokens(w, h, f):
        return (w // 16) * (h // 16) * ((f - 1) // 4 + 1)

    def take_seconds(self, w, h, f, steps):
        return self.base_take_s * (self.tokens(w, h, f) / self.tokens(720, 1280, 81)) ** 1.5 * steps / 25 * self.cal

    def job_seconds(self, r):
        return r["takes"] * self.take_seconds(r["width"], r["height"], r["frames"], r["steps"])

    def calibrate(self, actual, predicted):
        if predicted <= 0:
            return
        self.cal = max(0.05, min(20.0, 0.5 * self.cal + 0.5 * self.cal * actual / predicted))
        data = json.loads(self.cal_path.read_text()) if self.cal_path.exists() else {}
        data[self.backend.name] = self.cal
        self.cal_path.write_text(json.dumps(data))

    def eta(self, j):
        """Seconds until job j is done: the running job's remainder + every job ahead of it."""
        def remaining(x):
            r = x["request"]
            total = self.job_seconds(r)
            if x["state"] == "running":
                return max(0.0, total * (1 - x.get("progress", 0))) + (0 if self.backend.loaded else self.load_s)
            return total
        if j["state"] not in ("queued", "running"):
            return 0
        t = 0.0
        if j["state"] == "queued":
            if self.running:
                cur = self.jobs.get(self.running)
                t += remaining(cur) if cur else 0
            else:
                t += 0 if self.backend.loaded else self.load_s
            for jid in self.jobs.queued_ids():
                if jid == j["id"]:
                    break
                o = self.jobs.get(jid)
                if o and o["state"] == "queued":
                    t += remaining(o)
        return round(t + remaining(j))

    def public(self, j):
        j = dict(j)
        j["queue_position"] = self.jobs.position(j["id"])
        j["eta_seconds"] = self.eta(j)
        return j

    # ---- models / state ---------------------------------------------------------------------------
    def models(self):
        if self._models is None:
            self._models = self.backend.models()
        return self._models

    def state(self):
        if self.running:
            return "busy", None
        if not any(m["available"] for m in self.models()):
            return "unavailable", "No model is ready on this server (weights missing; see /v1/info checks)."
        why = self.backend.busy_reason()
        if why:
            return "unavailable", why
        return ("busy", None) if self.jobs.queue else ("idle", None)

    def keyframes(self):
        idx = self.keyframes_dir / "keyframes.json"
        meta = json.loads(idx.read_text()) if idx.exists() else {}
        out = []
        for p in sorted(self.keyframes_dir.glob("*.png")):
            name = p.stem
            if name.endswith("_sheet"):
                continue
            item, _, who = name.rpartition("_")
            if who not in ("pax", "polly"):
                item, who = name, "both"
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                continue
            out.append({"name": name, "action": item, "characters": who,
                        "setting": meta.get(name, {}).get("plate"), "width": w, "height": h})
        return out

    # ---- validation -----------------------------------------------------------------------------------
    def prompt_of(self, p):
        if bool(p.get("prompt")) == bool(p.get("guided")):
            raise PR.PromptError("give exactly one of 'prompt' (your own text) or 'guided' (the prompt builder)")
        if p.get("guided"):
            if not isinstance(p["guided"], dict):
                raise PR.PromptError("'guided' must be an object")
            return PR.compile_guided(p["guided"]), [], "guided"
        text = PR.clean(p["prompt"], "prompt", PR.MAX_PROMPT)
        if len(text) < 10:
            raise PR.PromptError("the prompt is too short")
        return text, PR.check(text), "custom"

    def validate(self, p):
        """params -> (request, warnings); raises HTTPException(422) with a readable message."""
        def bad(msg):
            raise HTTPException(422, msg)

        def whole(x):
            return isinstance(x, int) and not isinstance(x, bool)
        if not isinstance(p, dict):
            bad("params must be a JSON object")
        for f in ("model", "prompt", "size", "mode", "negative_extra", "start_preset"):
            if p.get(f) is not None and not isinstance(p[f], str):
                bad(f"{f} must be text")
        unknown = set(p) - {"model", "prompt", "guided", "size", "frames", "takes", "mode", "seed", "steps",
                            "negative_extra", "start_preset"}
        if unknown:
            bad(f"unknown parameters: {sorted(unknown)}")
        avail = {m["id"]: m for m in self.models()}
        model = p.get("model") or next((m["id"] for m in self.models() if m["available"]), None)
        if model not in avail:
            bad(f"model must be one of {sorted(avail)}")
        if not avail[model]["available"]:
            bad(f"model {model} is not ready on this server: missing {avail[model]['missing']}")
        size = p.get("size", "portrait_720")
        if size not in SIZES:
            bad(f"size must be one of {sorted(SIZES)}")
        frames = p.get("frames", 81)
        if not whole(frames) or frames not in FRAMES:
            bad(f"frames must be one of {list(FRAMES)}")
        takes = p.get("takes", 1)
        if not whole(takes) or not 1 <= takes <= MAX_TAKES:
            bad(f"takes must be a whole number from 1 to {MAX_TAKES}")
        mode = p.get("mode", "chain")
        if mode not in MODES:
            bad(f"mode must be one of {sorted(MODES)}")
        if mode == "first_last" and takes != 1:
            bad("mode first_last makes one take (start frame -> end frame); use takes=1")
        seed = p.get("seed")
        if seed is None:
            seed = secrets.randbelow(2 ** 31 - MAX_TAKES)
        if not whole(seed) or not 0 <= seed < 2 ** 31 - MAX_TAKES:
            bad("seed must be a whole number from 0 to 2147483644 (or null for a random one)")
        steps = p.get("steps", 25)
        if not whole(steps) or not 10 <= steps <= 50:
            bad("steps must be a whole number from 10 to 50 (25 is the tested setting)")
        try:
            prompt, warnings, source = self.prompt_of(p)
            neg_extra = PR.clean(p.get("negative_extra") or "", "negative_extra", PR.MAX_DETAILS)
        except PR.PromptError as e:
            bad(str(e))
        w, h, _ = SIZES[size]
        req = {"model": model, "size": size, "width": w, "height": h, "frames": frames, "takes": takes,
               "mode": mode, "seed": seed, "steps": steps, "prompt": prompt, "prompt_source": source,
               "guided": p.get("guided"), "negative_extra": neg_extra,
               "length_frames": takes * frames - (takes - 1)}
        req["length_seconds"] = round(req["length_frames"] / FPS, 2)
        return req, warnings

    def prepare_image(self, data, w, h, dst, what):
        """Upload bytes -> RGB PNG of exactly w x h at dst; refuses a shape that would lose the subject."""
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(422, f"{what} is larger than {MAX_IMAGE_BYTES // 2**20} MB")
        try:
            im = Image.open(io.BytesIO(data))
            im.verify()
            im = Image.open(io.BytesIO(data))
            if im.format not in ("PNG", "JPEG", "WEBP"):
                raise HTTPException(422, f"{what} must be PNG, JPEG or WebP (got {im.format})")
            im = im.convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
            raise HTTPException(422, f"{what} is not a readable image")
        iw, ih = im.size
        if min(iw, ih) < 256 or max(iw, ih) > 4096:
            raise HTTPException(422, f"{what} is {iw}×{ih}; it must be between 256 and 4096 pixels per side")
        ratio = (iw / ih) / (w / h)
        if max(ratio, 1 / ratio) > ASPECT_TOLERANCE:
            shape = lambda a, b: "portrait" if b > a else "landscape" if a > b else "square"  # noqa: E731
            raise HTTPException(422, f"{what} is {shape(iw, ih)} ({iw}×{ih}) but the output is {shape(w, h)} "
                                     f"({w}×{h}); cropping it would cut the characters off. Pick an output size "
                                     f"of the same shape, or crop the image first.")
        warn = None
        if ratio > 1:                               # wider than the output: trim the sides
            nw = round(ih * w / h)
            box = ((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih)
        else:
            nh = round(iw * h / w)
            box = (0, (ih - nh) // 2, iw, (ih - nh) // 2 + nh)
        lost = 1 - (box[2] - box[0]) * (box[3] - box[1]) / (iw * ih)
        if lost > 0.02:
            warn = f"{what} was cropped by {lost:.0%} to fit {w}×{h}."
        im.crop(box).resize((w, h), Image.LANCZOS).save(dst)
        return warn

    # ---- the worker -----------------------------------------------------------------------------------
    def worker(self):
        while not self.stop.is_set():
            jid = self.jobs.next(timeout=1.0)
            if jid:
                self.running = jid
                try:
                    self.run_job(jid)
                finally:
                    self.running = None
                    self.cancels.pop(jid, None)

    def run_job(self, jid):
        j = self.jobs.get(jid)
        r, d = j["request"], self.jobs.dir(jid)
        cancel = self.cancels.setdefault(jid, threading.Event())
        warnings = list(j.get("warnings", []))
        self.jobs.update(jid, state="running", stage="starting", started_utc=utc(), progress=0.0)
        log(f"job {jid}: start ({r['takes']} x {r['frames']} frames, {r['width']}x{r['height']}, {r['model']})")
        takes = []
        try:
            for k in range(1, r["takes"] + 1):
                if r["mode"] == "chain" and k > 1:
                    start = last_frame(takes[-1], d / f"take{k - 1}_last.png")
                else:
                    start = d / "start.png"
                end = {"loop": d / "start.png", "first_last": d / "end.png"}.get(r["mode"])
                t = {"model": r["model"], "prompt": r["prompt"], "start": str(start), "end": str(end) if end else None,
                     "frames": r["frames"], "seed": r["seed"] + k - 1, "steps": r["steps"], "width": r["width"],
                     "height": r["height"],
                     "neg": " ".join(x for x in (self.backend.default_negative(r["model"]), r["negative_extra"]) if x)}

                def on_event(ev, k=k):
                    st = ev.get("step", 0) / max(ev.get("steps", 1), 1)
                    stage = ev["stage"]
                    if stage == "generating":
                        stage = f"generating take {k} of {r['takes']}" if r["takes"] > 1 else "generating"
                    self.jobs.update(jid, stage=stage, take=k, step=ev.get("step"),
                                     progress=round(0.95 * ((k - 1) + st) / r["takes"], 4))
                warm = self.backend.loaded == r["model"]
                predicted = self.take_seconds(r["width"], r["height"], r["frames"], r["steps"])
                t0 = time.time()
                src = self.backend.take(t, on_event, cancel, timeout_s=max(1800.0, 3 * predicted + 1200))
                if warm:
                    self.calibrate(time.time() - t0, predicted)
                dst = d / f"take{k}.mp4"
                shutil.move(str(src), dst)
                n = ffprobe_frames(dst)
                if n != r["frames"]:
                    warnings.append(f"take {k} has {n} frames, {r['frames']} were asked for")
                takes.append(dst)
                log(f"job {jid}: take {k}/{r['takes']} done in {(time.time() - t0) / 60:.1f} min")
            self.jobs.update(jid, stage="joining takes" if len(takes) > 1 else "encoding", progress=0.96)
            video = d / "video.mp4"
            join_takes(takes, video)
            n = ffprobe_frames(video)
            if n != r["length_frames"]:
                warnings.append(f"the video has {n} frames, {r['length_frames']} were expected")
            if n <= 0:
                raise B.GenError("the finished video could not be read back")
            ffmpeg("-i", video, "-frames:v", "1", "-q:v", "3", d / "thumb.jpg")
            result = {"file": "video.mp4", "frames": n, "seconds": round(n / FPS, 2), "width": r["width"],
                      "height": r["height"], "size_mb": round(video.stat().st_size / 2**20, 2)}
            if self.a.upload and self.azure and self.azure.cc:
                self.jobs.update(jid, stage="saving to Azure", progress=0.98)
                try:
                    pre = f"inference/jobs/{jid}"
                    self.azure.put_file(f"{pre}/video.mp4", video, "video/mp4")
                    self.azure.put_file(f"{pre}/thumb.jpg", d / "thumb.jpg", "image/jpeg")
                    self.azure.put_file(f"{pre}/start.png", d / "start.png", "image/png")
                    result.update(blob=f"{pre}/video.mp4", thumb_blob=f"{pre}/thumb.jpg")
                except Exception as e:
                    warnings.append(f"not saved to Azure ({e}); download it from the server before it is switched off")
            elif self.a.upload:
                warnings.append(f"not saved to Azure ({self.azure.error if self.azure else 'off'})")
            j = self.jobs.update(jid, state="done", stage="done", progress=1.0, finished_utc=utc(),
                                 result=result, warnings=warnings)
            if result.get("blob"):
                try:
                    self.azure.put(f"inference/jobs/{jid}/job.json", json.dumps(j, indent=1), "application/json")
                except Exception as e:
                    log(f"job {jid}: job.json upload failed: {e}")
            log(f"job {jid}: done, {n} frames")
        except B.Cancelled:
            self.jobs.update(jid, state="cancelled", stage="cancelled", finished_utc=utc(), warnings=warnings)
            log(f"job {jid}: cancelled")
        except Exception as e:
            self.jobs.update(jid, state="failed", stage="failed", finished_utc=utc(), error=str(e), warnings=warnings)
            log(f"job {jid}: FAILED: {e}")

    # ---- registration + tunnel ------------------------------------------------------------------------
    def registry(self, state):
        if not (self.a.register and self.azure and self.azure.cc and self.public_url):
            return
        rec = {"service": "pudgy-inference", "version": VERSION, "state": state, "url": self.public_url,
               "backend": self.backend.name, "started_utc": self.started, "heartbeat_utc": utc(),
               "models": [m["id"] for m in self.models() if m["available"]],
               "gpu": self.backend.gpu().get("name")}
        if state == "stopped":
            rec["stopped_utc"] = utc()
        try:
            self.azure.put(REGISTRY_BLOB, json.dumps(rec, indent=1), "application/json")
        except Exception as e:
            log(f"registry write failed: {e}")

    def heartbeat(self):
        while not self.stop.is_set():
            self.registry("running")
            self.stop.wait(HEARTBEAT_S)

    def tunnel(self):
        exe = shutil.which("cloudflared") or "/opt/instance-tools/bin/cloudflared"
        while not self.stop.is_set():
            log("starting a Cloudflare quick tunnel")
            try:
                p = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.a.port}"],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            except OSError as e:
                log(f"cloudflared could not start ({e}); the server is reachable locally only")
                return
            self.tunnel_proc = p
            for line in p.stdout:
                m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
                if m and m.group(0) != self.public_url:
                    self.public_url = m.group(0)
                    (self.work / "public_url").write_text(self.public_url + "\n")
                    log(f"public address: {self.public_url}")
                    self.registry("running")
            p.wait()
            if not self.stop.is_set():
                log("the tunnel exited; restarting it in 10 s")
                self.stop.wait(10)


# ---------------------------------------------------------------------------------------------
class PromptBody(BaseModel):
    guided: dict | None = None
    prompt: str | None = None


def build_app(svc):
    key = os.environ.get("INFERENCE_API_KEY") or None

    @asynccontextmanager
    async def lifespan(app):
        threading.Thread(target=svc.worker, daemon=True, name="worker").start()
        if svc.a.tunnel:
            threading.Thread(target=svc.tunnel, daemon=True, name="tunnel").start()
        if svc.a.register:
            threading.Thread(target=svc.heartbeat, daemon=True, name="heartbeat").start()
        log(f"ready: backend {svc.backend.name}, models {[m['id'] for m in svc.models() if m['available']]}, "
            f"auth {'on' if key else 'OFF'}, port {svc.a.port}")
        yield
        svc.stop.set()
        svc.registry("stopped")
        for ev in svc.cancels.values():
            ev.set()
        svc.backend.stop()
        if svc.tunnel_proc and svc.tunnel_proc.poll() is None:
            svc.tunnel_proc.terminate()
        log("stopped")

    app = FastAPI(title="Pudgy Penguins video API", version=VERSION, lifespan=lifespan,
                  description="Generate Pudgy Penguins clips with the trained model. One GPU: jobs run one at a "
                              "time from a queue. Authorize with your API key, then POST /v1/jobs and poll "
                              "GET /v1/jobs/{id} until state is done.")
    header = APIKeyHeader(name="X-API-Key", auto_error=False)
    bearer = HTTPBearer(auto_error=False)

    def auth(k: str | None = Security(header), b: HTTPAuthorizationCredentials | None = Security(bearer)):
        if key is None:
            return
        given = k or (b.credentials if b else "")
        if not given or not secrets.compare_digest(given.encode(), key.encode()):
            raise HTTPException(401, "missing or wrong API key")

    def job_or_404(jid):
        j = svc.jobs.get(jid)
        if not j:
            raise HTTPException(404, f"no job {jid}")
        return j

    @app.get("/", include_in_schema=False)
    def root():
        return {"service": "pudgy-inference", "version": VERSION, "docs": "/docs", "health": "/v1/health"}

    @app.get("/v1/health", summary="Is the server up, and can it take a job? (no key needed)")
    def health():
        state, reason = svc.state()
        run = svc.jobs.get(svc.running) if svc.running else None
        return {"ok": True, "service": "pudgy-inference", "version": VERSION, "backend": svc.backend.name,
                "state": state, "reason": reason, "queue": len(svc.jobs.queue),
                "running": {"id": run["id"], "stage": run.get("stage"), "progress": run.get("progress", 0)} if run else None,
                "model_loaded": svc.backend.loaded, "gpu": svc.backend.gpu(), "started_utc": svc.started,
                "public_url": svc.public_url}

    @app.get("/v1/info", dependencies=[Depends(auth)], summary="Models, sizes, limits, prompt presets, start frames, checks")
    def info():
        return {
            "models": svc.models(),
            "sizes": {k: {"width": w, "height": h, "label": lab} for k, (w, h, lab) in SIZES.items()},
            "frames": list(FRAMES), "fps": FPS, "max_takes": MAX_TAKES, "modes": MODES,
            "limits": {"max_queue": MAX_QUEUE, "max_image_mb": MAX_IMAGE_BYTES // 2**20, "max_prompt": PR.MAX_PROMPT,
                       "steps": [10, 50], "aspect_tolerance": ASPECT_TOLERANCE},
            "defaults": {"size": "portrait_720", "frames": 81, "takes": 1, "mode": "chain", "steps": 25},
            "estimates": {k: {str(f): max(1, round(svc.take_seconds(w, h, f, 25))) for f in FRAMES}
                          for k, (w, h, _) in SIZES.items()},
            "load_seconds": round(svc.load_s), "model_loaded": svc.backend.loaded,
            "phone_embedding": svc.backend.name == "wan",
            "presets": PR.presets(), "keyframes": svc.keyframes(), "checks": svc.backend.checks(),
        }

    @app.post("/v1/prompt", dependencies=[Depends(auth)], summary="Preview the prompt the model will get, with advice")
    def prompt(body: PromptBody):
        try:
            text, warnings, _ = svc.prompt_of(body.model_dump())
        except PR.PromptError as e:
            raise HTTPException(422, str(e))
        return {"prompt": text, "warnings": warnings}

    @app.get("/v1/keyframes/{name}", dependencies=[Depends(auth)], summary="A preset start frame (PNG)")
    def keyframe(name: str):
        if name not in {k["name"] for k in svc.keyframes()}:
            raise HTTPException(404, f"no start frame {name}")
        return FileResponse(svc.keyframes_dir / f"{name}.png", media_type="image/png")

    @app.post("/v1/jobs", status_code=201, dependencies=[Depends(auth)], summary="Queue a video")
    async def submit(params: str = Form(..., description="JSON: see the README for every field"),
                     start_image: UploadFile | None = File(None), end_image: UploadFile | None = File(None)):
        state, reason = svc.state()
        if state == "unavailable":
            raise HTTPException(503, reason)
        if len(svc.jobs.queue) >= MAX_QUEUE:
            raise HTTPException(429, f"the queue is full ({MAX_QUEUE} jobs); try again when some have finished")
        try:
            p = json.loads(params)
        except json.JSONDecodeError as e:
            raise HTTPException(422, f"params is not valid JSON: {e}")
        req, warnings = svc.validate(p)
        preset = p.get("start_preset")
        if bool(preset) == bool(start_image):
            raise HTTPException(422, "give exactly one start frame: an upload (start_image) or start_preset")
        if req["mode"] == "first_last" and not end_image:
            raise HTTPException(422, "mode first_last needs an end_image")
        if end_image and req["mode"] != "first_last":
            raise HTTPException(422, "an end_image is only used with mode first_last")
        jid = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + secrets.token_hex(2)
        d = svc.jobs.dir(jid)
        d.mkdir(parents=True)
        try:
            if preset:
                if preset not in {k["name"] for k in svc.keyframes()}:
                    raise HTTPException(422, f"no preset start frame {preset!r}; see GET /v1/info keyframes")
                data = (svc.keyframes_dir / f"{preset}.png").read_bytes()
                req["start"] = {"source": "preset", "name": preset}
            else:
                data = await start_image.read()
                req["start"] = {"source": "upload", "name": Path(start_image.filename or "upload").name[:120]}
            w = svc.prepare_image(data, req["width"], req["height"], d / "start.png", "the start frame")
            warnings += [w] if w else []
            if end_image:
                w = svc.prepare_image(await end_image.read(), req["width"], req["height"], d / "end.png", "the end frame")
                warnings += [w] if w else []
                req["end"] = {"source": "upload", "name": Path(end_image.filename or "upload").name[:120]}
        except HTTPException:
            shutil.rmtree(d, ignore_errors=True)
            raise
        j = {"id": jid, "state": "queued", "stage": "queued", "created_utc": utc(), "started_utc": None,
             "finished_utc": None, "progress": 0.0, "take": 0, "takes": req["takes"], "step": None,
             "steps": req["steps"], "request": req, "warnings": warnings, "error": None, "result": None,
             "estimate_seconds": round(svc.job_seconds(req))}
        svc.jobs.add(j)
        log(f"job {jid}: queued ({req['prompt_source']} prompt, {req['size']}, {req['takes']} take(s))")
        return svc.public(j)

    @app.get("/v1/jobs", dependencies=[Depends(auth)], summary="Recent jobs, newest first")
    def jobs(limit: int = 20):
        return {"jobs": [svc.public(j) for j in svc.jobs.recent(max(1, min(limit, 100)))]}

    @app.get("/v1/jobs/{jid}", dependencies=[Depends(auth)], summary="One job: state, progress, ETA, result")
    def job(jid: str):
        return svc.public(job_or_404(jid))

    @app.get("/v1/jobs/{jid}/video", dependencies=[Depends(auth)], summary="The finished video (MP4)")
    def video(jid: str):
        j = job_or_404(jid)
        if j["state"] != "done":
            raise HTTPException(409, f"job {jid} is {j['state']}, not done")
        return FileResponse(svc.jobs.dir(jid) / "video.mp4", media_type="video/mp4", filename=f"pudgy_{jid}.mp4")

    @app.get("/v1/jobs/{jid}/thumbnail", dependencies=[Depends(auth)], summary="First frame of the finished video (JPEG)")
    def thumbnail(jid: str):
        job_or_404(jid)
        p = svc.jobs.dir(jid) / "thumb.jpg"
        if not p.exists():
            raise HTTPException(409, "no thumbnail yet")
        return FileResponse(p, media_type="image/jpeg")

    @app.delete("/v1/jobs/{jid}", dependencies=[Depends(auth)], summary="Cancel a queued or running job")
    def cancel(jid: str):
        j = job_or_404(jid)
        if j["state"] == "queued":
            svc.jobs.remove_from_queue(jid)
            return svc.public(svc.jobs.update(jid, state="cancelled", stage="cancelled", finished_utc=utc()))
        if j["state"] == "running":
            svc.cancels.setdefault(jid, threading.Event()).set()
            return svc.public(svc.jobs.update(jid, stage="cancelling"))
        raise HTTPException(409, f"job {jid} is already {j['state']}")

    return app


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("wan", "mock"), default="wan")
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 behind the tunnel; 0.0.0.0 to expose the port")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ws", default="/workspace", help="the box workspace (models, wan_output, Pudgy)")
    ap.add_argument("--work", default=None, help="jobs, logs, keyframes (default <ws>/infer_api)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--models", default="v8merged", help="comma list of common_v10 configurations, e.g. v8merged,v10@250")
    ap.add_argument("--keyframes", default=None, help="preset start frames (default <work>/keyframes)")
    ap.add_argument("--tunnel", action="store_true", help="open a Cloudflare quick tunnel and use it as the public address")
    ap.add_argument("--public-url", default=None)
    ap.add_argument("--register", action="store_true", help="write the address + heartbeat to Azure for the dashboard")
    ap.add_argument("--upload", action="store_true", help="save finished videos to the dashboard's Azure container")
    ap.add_argument("--no-auth", action="store_true", help="no API key (local testing only)")
    ap.add_argument("--env-file", default=None, help="NAME=value lines to read (default <ws>/Pudgy/.env)")
    ap.add_argument("--mock-step-seconds", type=float, default=0.15)
    a = ap.parse_args(argv)
    a.work = a.work or str(Path(a.ws) / "infer_api")
    return a


def main():
    a = parse_args()
    read_env_file(a.env_file or Path(a.ws) / "Pudgy" / ".env")
    if a.no_auth:
        os.environ.pop("INFERENCE_API_KEY", None)
    elif not os.environ.get("INFERENCE_API_KEY"):
        sys.exit("INFERENCE_API_KEY is not set (environment or the repo .env). Make one with\n"
                 "  python -c 'import secrets; print(secrets.token_urlsafe(32))'\n"
                 "and give the same value to the dashboard (App Service setting INFERENCE_API_KEY).")
    if a.register and not (a.tunnel or a.public_url):
        sys.exit("--register needs a public address: --tunnel or --public-url")
    if a.tunnel and a.host != "127.0.0.1":
        log("note: --tunnel with --host other than 127.0.0.1 also exposes the port directly")
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} is not on PATH")
    svc = Service(a)
    uvicorn.run(build_app(svc), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
