"""Generation backends for the inference API.

WanBackend  the production model through musubi's --interactive mode: ONE model load, then one
            prompt line per take on stdin (longform_client/stream.py's pattern). The command comes
            from eval_v10/common_v10.musubi_cmd - the same two experts, LoRAs, sampler settings and
            E5 phone embedding (applied whenever the prompt holds the registry phone phrase) as the
            v10 evaluation - built by a one-off call under the trainer venv, so this process never
            imports torch. musubi exits on any error in interactive mode (only Ctrl-C/EOF are
            caught), so a dead process = a failed take, and the next job starts a fresh one.
MockBackend placeholder videos from the start frame, no GPU: for tests and for trying the dashboard.

Both expose: checks(), models(), gpu(), busy_reason(), loaded, take(t, on_event, cancel, timeout_s), stop().
A take t = {prompt, start, end?, frames, seed, steps, width, height, neg?}.
"""
import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

import prompts as PR

PROGRESS = re.compile(r"(\d+)/(\d+) \[")
LOADING = re.compile(r"Loading (DiT|text encoder|VAE|T5|CLIP|LoRA)|load(ing)? (lora|weights)", re.I)
MODEL_LABELS = {"v8merged": "v8 production (v8 merged low-noise + golden high-noise, phone E5)"}
GPU_FREE_MB = 6000          # more than this in use before our model loads = someone else's job (training)


class GenError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


def model_label(cfg):
    if cfg in MODEL_LABELS:
        return MODEL_LABELS[cfg]
    if cfg.startswith("v10@"):
        return f"v10 candidate, step {cfg[4:]} (golden + v10 high-noise LoRA, v8 merged low-noise, phone E5)"
    return cfg


def take_line(t):
    """musubi's interactive line (common_v10.job_line + size and steps). Every user string in it
    has passed prompts.clean, so it holds no ' --' and no newline."""
    line = (f"{t['prompt']} --i {t['start']} --f {t['frames']} --d {t['seed']} "
            f"--w {t['width']} --h {t['height']} --s {t['steps']}")
    if t.get("neg"):
        line += f" --n {t['neg']}"
    if t.get("end"):
        line += f" --ei {t['end']}"
    return line


def nvidia_smi(index):
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        name, used, total, util = [x.strip() for x in out.strip().split(",")]
        return {"name": name, "memory_used_mb": int(used), "memory_total_mb": int(total), "utilization": int(util)}
    except Exception as e:                                   # no driver, no GPU, odd output
        return {"error": f"nvidia-smi failed: {e}"}


class WanBackend:
    name = "wan"

    def __init__(self, ws, work, gpu="0", models=("v8merged",)):
        self.ws, self.work, self.gpu = Path(ws), Path(work), str(gpu)
        self.repo = self.ws / "Pudgy"
        self.pyv = self.repo / ".venv-wan" / "bin" / "python"
        self.eval10 = self.repo / "finetune" / "wan" / "eval_v10"
        self.config_ids = list(models)
        self.raw = self.work / "raw"
        self.log = self.work / "logs" / "generator.log"
        self.proc, self.loaded, self.lines = None, None, None
        self._desc = {}
        self._gpu_cache = (0.0, None)

    # ---- set-up -----------------------------------------------------------------------------
    def describe(self, cfg):
        """The musubi command, missing weight files and Wan's default negative prompt for a model
        configuration, from common_v10 under the trainer venv (cached)."""
        if cfg in self._desc:
            return self._desc[cfg]
        code = ("import json, sys; sys.path.insert(0, sys.argv[3]); import common_v10 as C; "
                "sys.path.insert(0, str(C.C8.MUSUBI / 'src')); "
                "from musubi_tuner.wan.configs.shared_config import wan_shared_cfg as W; "
                "miss = C.missing_files(sys.argv[1]) + ([] if C.E5.exists() else [str(C.E5)]); "
                "print('DESC ' + json.dumps({'cmd': C.musubi_cmd(sys.argv[1], sys.argv[2], (1280, 720), True), "
                "'missing': miss, 'cwd': str(C.C8.MUSUBI), 'musubi': str(C.C8.MUSUBI), "
                "'neg': W.sample_neg_prompt}))")
        try:
            # CUDA_VISIBLE_DEVICES: common_v10 defaults it to GPU 1 at import; only a command is built here
            r = subprocess.run([str(self.pyv), "-c", code, cfg, str(self.raw), str(self.eval10)], capture_output=True,
                               text=True, timeout=180, env=dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu))
            line = next((x for x in r.stdout.splitlines() if x.startswith("DESC ")), None)
            d = json.loads(line[5:]) if line else {"error": (r.stderr or r.stdout).strip()[-600:]}
        except Exception as e:
            d = {"error": str(e)}
        self._desc[cfg] = d
        return d

    def models(self):
        out = []
        for cfg in self.config_ids:
            d = self.describe(cfg)
            miss = d.get("missing", []) if "error" not in d else [d["error"]]
            out.append({"id": cfg, "label": model_label(cfg), "available": not miss, "missing": miss})
        return out

    def checks(self):
        c = [{"name": "trainer venv", "ok": self.pyv.exists(), "detail": str(self.pyv)},
             {"name": "eval_v10 code", "ok": (self.eval10 / "common_v10.py").exists(), "detail": str(self.eval10)}]
        for tool in ("ffmpeg", "ffprobe"):
            ok = subprocess.run(["which", tool], capture_output=True).returncode == 0
            c.append({"name": tool, "ok": ok, "detail": "found" if ok else "not on PATH"})
        for m in self.models():
            c.append({"name": f"model {m['id']}", "ok": m["available"],
                      "detail": "weights present" if m["available"] else "missing: " + ", ".join(m["missing"])[:400]})
        g = self.gpu()
        c.append({"name": "GPU", "ok": "error" not in g, "detail": g.get("error") or f"{g['name']}, "
                  f"{g['memory_used_mb'] / 1024:.0f} of {g['memory_total_mb'] / 1024:.0f} GB in use"})
        return c

    def gpu(self):
        t, g = self._gpu_cache
        if g is None or time.time() - t > 10:
            g = nvidia_smi(self.gpu_index())
            self._gpu_cache = (time.time(), g)
        return g

    def gpu_index(self):
        return self.gpu

    def busy_reason(self):
        """Why a new model load would not fit, or None. Once our generator is up, it is the user."""
        if self.alive():
            return None
        g = self.gpu()
        if "error" in g:
            return g["error"]
        if g["memory_used_mb"] > GPU_FREE_MB:
            return (f"The GPU is in use by another job ({g['memory_used_mb'] / 1024:.0f} GB taken, probably a "
                    f"training run); generation would run out of memory. It becomes available when that job ends.")
        return None

    # ---- the generator process ----------------------------------------------------------------
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _start(self, cfg):
        d = self.describe(cfg)
        if "error" in d:
            raise GenError(f"could not prepare model {cfg}: {d['error']}")
        if d["missing"]:
            raise GenError(f"model {cfg} is missing weight files: {', '.join(d['missing'])}")
        why = self.busy_reason()
        if why:
            raise GenError(why)
        self.raw.mkdir(parents=True, exist_ok=True)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1",
                   MUSUBI=d["musubi"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        with open(self.log, "a") as f:
            f.write(f"\n==== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} start {cfg}\n")
        self.proc = subprocess.Popen(d["cmd"], cwd=d["cwd"], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.loaded = cfg
        self.lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self.lines), daemon=True).start()

    def _pump(self, proc, q):
        with open(self.log, "a") as log:
            for line in proc.stdout:          # universal newlines: tqdm's \r updates arrive as lines
                log.write(line)
                log.flush()
                q.put(line)
        q.put(None)

    def default_negative(self, cfg):
        neg = (self.describe(cfg).get("neg") or "").replace("\n", " ")
        return neg + PR.NEG_EXTRA if neg and "--" not in neg else ""

    def take(self, t, on_event, cancel, timeout_s):
        cfg = t["model"]
        if self.alive() and self.loaded != cfg:
            on_event({"stage": f"switching model to {cfg}"})
            self.stop()
        if not self.alive():
            on_event({"stage": "loading model"})
            self._start(cfg)
        self.proc.stdin.write(take_line(t) + "\n")
        self.proc.stdin.flush()
        t0, tail = time.time(), []
        while True:
            if cancel.is_set():
                self.stop()
                raise Cancelled("cancelled")
            if time.time() - t0 > timeout_s:
                self.stop()
                raise GenError(f"the take took longer than {timeout_s / 60:.0f} min and was stopped")
            try:
                line = self.lines.get(timeout=1)
            except queue.Empty:
                continue
            if line is None:
                code = self.proc.wait()
                self.proc, self.loaded = None, None
                raise GenError(f"the generator stopped (exit {code}): " + " | ".join(x.strip() for x in tail[-4:]))
            tail = (tail + [line])[-40:]
            if "Video saved to:" in line:
                src = Path(line.split("Video saved to:", 1)[1].strip())
                return src if src.is_absolute() else Path(self.describe(cfg)["cwd"]) / src
            m = PROGRESS.search(line)
            if m and int(m.group(2)) == t["steps"]:
                on_event({"stage": "generating", "step": int(m.group(1)), "steps": t["steps"]})
            elif LOADING.search(line):
                on_event({"stage": "loading model"})

    def stop(self):
        p, self.proc, self.loaded = self.proc, None, None
        if p is None or p.poll() is not None:
            return
        try:
            p.stdin.close()
        except Exception:
            pass
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()


class MockBackend:
    """Placeholder videos: the start frame with a slow push-in, blending into the end frame when one
    is given, and a moving marker so a chain's takes are visibly different. A prompt containing
    MOCK_FAIL fails the take; MOCK_GPU_BUSY=1 in the environment reports the GPU as taken."""
    name = "mock"

    def __init__(self, work, step_seconds=0.15, load_seconds=1.5):
        self.work = Path(work)
        self.step_seconds, self.load_seconds = step_seconds, load_seconds
        self.loaded = None

    def models(self):
        return [{"id": "v8merged", "label": "mock (placeholder videos, no GPU)", "available": True, "missing": []}]

    def checks(self):
        return [{"name": "backend", "ok": True, "detail": "mock backend: placeholder videos, no GPU"}]

    def gpu(self):
        return {"name": "mock (no GPU)", "memory_used_mb": 0, "memory_total_mb": 0, "utilization": 0}

    def busy_reason(self):
        return "The GPU is in use by another job (mock)." if os.environ.get("MOCK_GPU_BUSY") else None

    def default_negative(self, cfg):
        return "mock negative" + PR.NEG_EXTRA

    def take(self, t, on_event, cancel, timeout_s):
        from PIL import Image, ImageDraw
        if self.loaded != t["model"]:
            why = self.busy_reason()
            if why:
                raise GenError(why)
            on_event({"stage": "loading model"})
            time.sleep(self.load_seconds)
            self.loaded = t["model"]
        for s in range(1, t["steps"] + 1):
            if cancel.is_set():
                raise Cancelled("cancelled")
            time.sleep(self.step_seconds)
            on_event({"stage": "generating", "step": s, "steps": t["steps"]})
        if "MOCK_FAIL" in t["prompt"]:
            raise GenError("mock failure requested by the prompt")
        w, h, n = t["width"], t["height"], t["frames"]
        a = Image.open(t["start"]).convert("RGB").resize((w, h))
        b = Image.open(t["end"]).convert("RGB").resize((w, h)) if t.get("end") else None
        out = self.work / "raw" / f"mock_{time.time_ns()}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        enc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{w}x{h}", "-r", "24", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                str(out)], stdin=subprocess.PIPE)
        for i in range(n):
            u = i / max(n - 1, 1)
            z = 1 + 0.06 * u
            f = a.resize((int(w * z), int(h * z))).crop((int(w * (z - 1) / 2), int(h * (z - 1) / 2),
                                                          int(w * (z - 1) / 2) + w, int(h * (z - 1) / 2) + h))
            if b is not None:
                f = Image.blend(f, b, u)
            d = ImageDraw.Draw(f)
            x = int(20 + (w - 60) * u)
            d.rectangle([x, 16, x + 24, 40], fill=(255, 90, 90))
            d.text((20, h - 30), f"MOCK seed {t['seed']}  frame {i + 1}/{n}", fill=(20, 20, 20))
            enc.stdin.write(f.tobytes())
        enc.stdin.close()
        if enc.wait() != 0:
            raise GenError("mock encoder failed")
        return out

    def stop(self):
        self.loaded = None
