"""CPU tests for the inference API: the mock backend makes real (placeholder) MP4s through ffmpeg,
so the whole path - validation, queue, takes, chaining, joining, frame checks - runs without a GPU.

    <venv with requirements.txt + httpx pytest>/bin/python -m pytest finetune/wan/infer_api/tests -q
"""
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import prompts as PR  # noqa: E402
import server as S  # noqa: E402

KEY = "test-key-123"
H = {"X-API-Key": KEY}


def png(w, h, colour=(120, 170, 230)):
    b = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(b, "PNG")
    return b.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    kf = tmp_path / "keyframes"
    kf.mkdir()
    (kf / "waving_pax.png").write_bytes(png(720, 1280))
    (kf / "hug.png").write_bytes(png(720, 1280, (240, 180, 200)))
    (kf / "keyframes.json").write_text(json.dumps({"waving_pax": {"plate": "door"}, "hug": {"plate": "sofa"}}))
    monkeypatch.setenv("INFERENCE_API_KEY", KEY)
    monkeypatch.delenv("MOCK_GPU_BUSY", raising=False)
    a = S.parse_args(["--backend", "mock", "--work", str(tmp_path / "work"), "--keyframes", str(kf),
                      "--mock-step-seconds", "0.002"])
    svc = S.Service(a)
    with TestClient(S.build_app(svc)) as c:
        c.svc = svc
        yield c


def submit(c, params, files=None):
    return c.post("/v1/jobs", headers=H, data={"params": json.dumps(params)}, files=files or {})


def wait(c, jid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/v1/jobs/{jid}", headers=H).json()
        if j["state"] not in ("queued", "running"):
            return j
        time.sleep(0.1)
    raise AssertionError(f"job {jid} still {j['state']} after {timeout} s")


GUIDED = {"characters": "pax", "action": "waving", "setting": "door"}


# ---- auth + discovery ---------------------------------------------------------------------------
def test_health_is_public_and_the_rest_needs_the_key(client):
    h = client.get("/v1/health").json()
    assert h["ok"] and h["state"] == "idle" and h["backend"] == "mock"
    assert client.get("/v1/info").status_code == 401
    assert client.get("/v1/info", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/info", headers=H).status_code == 200
    assert client.get("/v1/info", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200
    assert client.get("/v1/jobs").status_code == 401


def test_info_lists_presets_keyframes_and_estimates(client):
    i = client.get("/v1/info", headers=H).json()
    assert {k["name"] for k in i["keyframes"]} == {"waving_pax", "hug"}
    hug = next(k for k in i["keyframes"] if k["name"] == "hug")
    assert hug["characters"] == "both" and hug["setting"] == "sofa"
    assert {"walking", "hug", "phone_texting_sitting"} <= set(i["presets"]["actions"])
    assert all(s >= 1 for e in i["estimates"].values() for s in e.values())
    ts = client.svc.take_seconds
    assert ts(720, 1280, 81, 25) > ts(480, 848, 81, 25) > ts(480, 848, 33, 25) > 0
    assert client.get("/v1/keyframes/hug", headers=H).headers["content-type"] == "image/png"
    assert client.get("/v1/keyframes/..%2Fjob", headers=H).status_code == 404


# ---- prompts --------------------------------------------------------------------------------------
def test_guided_prompt_is_the_long_form_wording(client):
    v = PR.vocab()
    want = (f"{v['style']}{v['anchors']['Pax']}, {PR.vocab()['actions']['walking']['clause']}; walk cycle"
            f"{PR.LOOP_SUFFIX}; medium shot, {v['cam']}; {v['sets']['sofa']}.")
    r = client.post("/v1/prompt", headers=H, json={"guided": {"characters": "pax", "action": "walking", "setting": "sofa"}})
    assert r.status_code == 200 and r.json()["prompt"] == want and r.json()["warnings"] == []


def test_prompt_injection_and_line_breaks_are_refused(client):
    for text in ("Pax waves --i /etc/passwd", "Pax waves\n--d 1", "Pax waves\rhello", "x" * 2000):
        r = client.post("/v1/prompt", headers=H, json={"prompt": text})
        assert r.status_code == 422, text
    r = client.post("/v1/prompt", headers=H, json={"guided": dict(GUIDED, details="happy --cn /tmp/x")})
    assert r.status_code == 422


def test_custom_prompt_gets_advice(client):
    r = client.post("/v1/prompt", headers=H, json={"prompt": "Pax waves at the camera in a close-up"})
    w = " ".join(r.json()["warnings"])
    assert "style prefix" in w and "trained wording" in w and "Close-ups" in w


# ---- jobs -----------------------------------------------------------------------------------------
def test_chain_of_two_takes_from_a_preset(client):
    r = submit(client, {"guided": GUIDED, "start_preset": "waving_pax", "size": "portrait_480", "frames": 33,
                        "takes": 2, "seed": 7})
    assert r.status_code == 201, r.text
    j = wait(client, r.json()["id"])
    assert j["state"] == "done", j
    assert j["result"]["frames"] == 65 == j["request"]["length_frames"]          # 2 x 33 - 1 seam frame
    v = client.get(f"/v1/jobs/{j['id']}/video", headers=H)
    assert v.status_code == 200 and v.headers["content-type"] == "video/mp4"
    d = client.svc.jobs.dir(j["id"])
    assert S.ffprobe_frames(d / "take1.mp4") == 33 and (d / "take1_last.png").exists()
    assert client.get(f"/v1/jobs/{j['id']}/thumbnail", headers=H).status_code == 200
    assert j["warnings"] == []


def test_loop_and_first_last_modes(client):
    up = {"start_image": ("a.png", png(480, 848), "image/png")}
    j = wait(client, submit(client, {"guided": GUIDED, "size": "portrait_480", "frames": 33, "takes": 2,
                                     "mode": "loop"}, up).json()["id"])
    assert j["state"] == "done" and j["result"]["frames"] == 65
    files = {"start_image": ("a.png", png(480, 848), "image/png"),
             "end_image": ("b.png", png(480, 848, (250, 250, 250)), "image/png")}
    j = wait(client, submit(client, {"guided": GUIDED, "size": "portrait_480", "frames": 33,
                                     "mode": "first_last"}, files).json()["id"])
    assert j["state"] == "done" and j["result"]["frames"] == 33


@pytest.mark.parametrize("params,files,needle", [
    ({"frames": 80}, None, "frames"),
    ({"takes": 4}, None, "takes"),
    ({"takes": True}, None, "takes"),
    ({"steps": 5}, None, "steps"),
    ({"seed": -1}, None, "seed"),
    ({"size": "8k"}, None, "size"),
    ({"model": "v99"}, None, "model"),
    ({"colour": "red"}, None, "unknown"),
    ({"mode": "first_last", "takes": 2}, None, "takes=1"),
    ({"mode": "first_last"}, None, "end_image"),
    ({"prompt": "Pax waves hello there"}, None, "exactly one"),          # prompt + guided
    ({"start_preset": "nope"}, None, "preset"),
    ({"start_preset": None}, None, "start frame"),                        # no start at all
    ({"start_preset": ["x"]}, None, "text"),
])
def test_bad_requests_are_refused_with_a_reason(client, params, files, needle):
    p = {"guided": GUIDED, "start_preset": "waving_pax", **params}
    p = {k: v for k, v in p.items() if v is not None}
    r = submit(client, p, files)
    assert r.status_code == 422 and needle in r.json()["detail"], r.text


def test_wrong_shape_start_frame_is_refused_and_small_differences_are_cropped(client):
    r = submit(client, {"guided": GUIDED, "size": "portrait_720"},
               {"start_image": ("wide.png", png(1280, 720), "image/png")})
    assert r.status_code == 422 and "landscape" in r.json()["detail"]
    r = submit(client, {"guided": GUIDED, "size": "portrait_480", "frames": 33},
               {"start_image": ("near.png", png(500, 848), "image/png")})          # 4% wider: cropped
    assert r.status_code == 201 and any("cropped" in w for w in r.json()["warnings"])
    r = submit(client, {"guided": GUIDED, "size": "portrait_480"},
               {"start_image": ("t.txt", b"not an image", "text/plain")})
    assert r.status_code == 422 and "readable image" in r.json()["detail"]


def test_failure_is_reported_and_the_next_job_still_runs(client):
    v = PR.vocab()
    bad = f"{v['style']}{v['anchors']['Pax']}, waving MOCK_FAIL"
    j = wait(client, submit(client, {"prompt": bad, "start_preset": "waving_pax", "size": "portrait_480",
                                     "frames": 33}).json()["id"])
    assert j["state"] == "failed" and "mock failure" in j["error"]
    j = wait(client, submit(client, {"guided": GUIDED, "start_preset": "waving_pax", "size": "portrait_480",
                                     "frames": 33}).json()["id"])
    assert j["state"] == "done"


def test_cancel_a_queued_job(client):
    client.svc.backend.step_seconds = 0.05
    first = submit(client, {"guided": GUIDED, "start_preset": "waving_pax", "frames": 33, "takes": 3}).json()
    second = submit(client, {"guided": GUIDED, "start_preset": "waving_pax", "frames": 33}).json()
    assert second["queue_position"] >= 1 and second["eta_seconds"] > 0
    r = client.delete(f"/v1/jobs/{second['id']}", headers=H)
    assert r.status_code == 200 and r.json()["state"] == "cancelled"
    assert client.delete(f"/v1/jobs/{second['id']}", headers=H).status_code == 409
    client.delete(f"/v1/jobs/{first['id']}", headers=H)
    assert wait(client, first["id"])["state"] in ("cancelled", "done")


def test_gpu_taken_by_another_job_makes_the_server_unavailable(client, monkeypatch):
    monkeypatch.setenv("MOCK_GPU_BUSY", "1")
    h = client.get("/v1/health").json()
    assert h["state"] == "unavailable" and "GPU" in h["reason"]
    r = submit(client, {"guided": GUIDED, "start_preset": "waving_pax"})
    assert r.status_code == 503


def test_a_restart_marks_running_jobs_interrupted_and_requeues_waiting_ones(tmp_path):
    root = tmp_path / "jobs"
    for jid, state in (("20260101-000000-aaaa", "running"), ("20260101-000001-bbbb", "queued")):
        (root / jid).mkdir(parents=True)
        (root / jid / "job.json").write_text(json.dumps({"id": jid, "state": state}))
    jobs = S.Jobs(root)
    assert jobs.get("20260101-000000-aaaa")["state"] == "interrupted"
    assert jobs.queued_ids() == ["20260101-000001-bbbb"]


def test_env_file_is_read_not_executed(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("bare_token_line\nimport os\nos.environ['X'] = 'y'\n  indented=1\n"
                 "INFERENCE_TEST_A=\"quoted value\"\nexport INFERENCE_TEST_B=plain;with=semicolons\n")
    monkeypatch.delenv("INFERENCE_TEST_A", raising=False)
    monkeypatch.delenv("INFERENCE_TEST_B", raising=False)
    S.read_env_file(f)
    assert os.environ["INFERENCE_TEST_A"] == "quoted value"
    assert os.environ["INFERENCE_TEST_B"] == "plain;with=semicolons"
    assert "indented" not in os.environ


def test_vocab_sources_still_parse():
    v = PR.vocab()
    assert v["phone_phrase"].startswith("the iFin phone") and len(v["sets"]) == 4
    assert subprocess.run([sys.executable, "-c", "import prompts"], cwd=HERE.parent).returncode == 0
