"""finetune/wan/cache_mark_v10.py on synthetic musubi caches (fixtures.make_ws), plus musubi v0.3.4's
real training-time dataset pipeline reading the marked files back into batches."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Value
from pathlib import Path

import common
from common import check
import torch
from fixtures import PH, PH_N, make_ws, write_te
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SCRIPT = common.FW / "cache_mark_v10.py"
RUN_PY = common.HERE / "run_py.py"


def run(*args):
    r = subprocess.run([sys.executable, "-B", str(RUN_PY), str(SCRIPT), *args], capture_output=True, text=True,
                       env=os.environ.copy())
    return r.returncode, r.stdout + r.stderr


def meta(p):
    with safe_open(str(p), "pt") as f:
        return f.metadata() or {}


def musubi_batches(cfg):
    """musubi's own training-time dataset path (trainer_base: BlueprintGenerator -> dataset group)."""
    bp = BlueprintGenerator(ConfigSanitizer()).generate(config_utils.load_user_config(str(cfg)), argparse.Namespace(),
                                                        architecture="wan")
    ep = Value("i", 0)
    group = config_utils.generate_dataset_group_by_blueprint(bp.dataset_group, training=True, shared_epoch=ep)
    return [group[i] for i in range(len(group))]


T = Path(tempfile.mkdtemp(prefix="mark_", dir=common.TMP))
print("== mark / check / idempotence")
cfg, cache, e5 = make_ws(T / "ws")
te0 = sorted(cache.glob("scene_f21/*_wan_te.safetensors"))[0]
before = load_file(te0)
md_before = meta(te0)
rc, out = run("check", "--config", str(cfg), "--asset", str(e5))
check(rc == 1 and "not marked" in out, "check on an unmarked cache fails")
rc, out = run("mark", "--config", str(cfg), "--asset", str(e5), "--dry-run")
check(rc == 0 and "would write 11" in out and load_file(te0).keys() == before.keys(), "dry run writes nothing")
rc, out = run("mark", "--config", str(cfg), "--asset", str(e5), "--cache-root", str(cache))
print("   " + "\n   ".join(out.strip().splitlines()[-5:]))
check(rc == 0 and "wrote 11" in out and "CACHE_MARK_CHECK OK replay=3 scene=8 phone=3" in out, "mark: 11 files, check OK")
after = load_file(te0)
check(torch.equal(after["varlen_t5_bfloat16"], before["varlen_t5_bfloat16"]) and after["varlen_t5_bfloat16"].dtype == torch.bfloat16,
      "T5 rows bit-identical after marking (bf16 kept)")
md = meta(te0)
check(all(md[k] == v for k, v in md_before.items()) and md["pudgy_mark"] == "v10", "musubi's metadata kept, marks added")
rc, out = run("mark", "--config", str(cfg), "--asset", str(e5))
check(rc == 0 and "wrote 0, unchanged 11" in out, "second mark is a no-op (idempotent)")
check(not list(cache.rglob("*.marktmp")), "no temp files left")

print("== musubi v0.3.4's dataset pipeline reads the marks into the batch")
kinds, bad = [], []
for b in musubi_batches(cfg):
    ok = isinstance(b.get("preserve"), list) and isinstance(b.get("assetspan"), list) and isinstance(b.get("t5"), list)
    p, (s, e) = int(b["preserve"][0][0]), b["assetspan"][0].tolist()
    kinds.append("replay" if p else ("phone" if e > s else "plain"))
    if not (ok and len(b["preserve"]) == 1 and b["latents"].shape[0] == 1 and e - s in (0, PH_N)):
        bad.append(sorted(b))
check(not bad, f"every batch carries preserve/assetspan/t5 as per-sample lists, spans of {PH_N} tokens ({bad})")
check(sorted(kinds) == sorted(["replay"] * 3 + ["phone"] * 3 + ["plain"] * 5), f"11 batches: {sorted(kinds)}")

print("== a musubi T5 re-cache keeps the marks; a changed caption is caught")
te_phone = next(p for p in sorted(cache.glob("scene_f81/*_wan_te.safetensors")) if meta(p).get("pudgy_span", "0,0") != "0,0")
cap = meta(te_phone)["caption1"]
write_te(te_phone, "x.mp4", cap)                                  # musubi's writer, merge_existing=True
rc, out = run("check", "--config", str(cfg), "--asset", str(e5))
check(rc == 0 and "varlen_preserve" in load_file(te_phone), "same-caption re-cache: musubi merge_existing keeps the marks, check OK")
new_cap = cap + " Extra words."
write_te(te_phone, "x.mp4", new_cap)
jl = T / "ws/data_v10/dataset_v10_scene_f81.workspace.jsonl"
jl.write_text(jl.read_text().replace(json.dumps(cap)[1:-1], json.dumps(new_cap)[1:-1]))
rc, out = run("check", "--config", str(cfg), "--asset", str(e5))
check(rc == 1 and "marked for another caption" in out, "changed caption + re-cache: check fails until re-marked")
rc, out = run("mark", "--config", str(cfg), "--asset", str(e5))
check(rc == 0 and "wrote 1," in out, "re-mark fixes exactly that file")

print("== smoke set")
smoke = T / "ws/wan_cache/smoke"
rc, out = run("smoke", "--config", str(cfg), "--asset", str(e5), "--cache-root", str(cache), "--out", str(smoke))
print("   " + "\n   ".join(out.strip().splitlines()[-4:]))
st = smoke / "smoke.toml"
check(rc == 0 and st.exists(), "smoke set built")
rc2, out2 = run("check", "--config", str(st), "--asset", str(e5), "--cache-root", str(smoke))
check(rc2 == 0 and "CACHE_MARK_CHECK OK replay=1 scene=2 phone=1" in out2, "smoke set passes the check (1 replay, 1 phone, 1 plain)")
check("replay_f57" in out and "scene_f81" in out.split("smoke phone")[1].split("\n")[0],
      "heaviest replay (f57 1024^2) and an f81 phone window picked")
sb = musubi_batches(st)
check(len(sb) == 3 and sorted(int(b["preserve"][0][0]) for b in sb) == [0, 0, 1], "musubi sees 3 batches, 1 replay-marked")
rc, out = run("smoke", "--config", str(cfg), "--asset", str(e5), "--out", str(smoke))
check(rc == 0, "smoke rebuild over its own dir works")
(T / "ws/notsmoke").mkdir()
rc, out = run("smoke", "--config", str(cfg), "--asset", str(e5), "--out", str(T / "ws/notsmoke"))
check(rc != 0 and "refusing" in out, "smoke refuses to delete a dir it did not make")

print("== inconsistencies -> exit 1, nothing written")
root = T / "bad_replay"
c, ca, e = make_ws(root)
jl = root / "data_v10/dataset_v10_replay_f57.workspace.jsonl"
r0 = json.loads(jl.read_text().splitlines()[0])
badcap = r0["caption"].replace("waving one flipper", f"holding {PH}, and waving")
te = ca / f"replay_f57/{Path(r0['video_path']).stem}_wan_te.safetensors"
os.remove(te)
write_te(te, r0["video_path"], badcap)
jl.write_text(jl.read_text().replace(json.dumps(r0["caption"])[1:-1], json.dumps(badcap)[1:-1]))
rc, out = run("mark", "--config", str(c), "--asset", str(e))
check(rc == 1 and "replay caption names the phone" in out and "nothing written" in out
      and not any("varlen_preserve" in load_file(p) for p in ca.rglob("*_te.safetensors")), "replay caption with the phone phrase")

root = T / "missing_te"
c, ca, e = make_ws(root)
os.remove(sorted(ca.glob("scene_f21/*_wan_te.safetensors"))[0])
rc, out = run("mark", "--config", str(c), "--asset", str(e))
check(rc == 1 and "_te file missing" in out, "missing _te file")

root = T / "stale_latent"
c, ca, e = make_ws(root)
src = sorted(ca.glob("scene_f21/*_wan.safetensors"))[0]
shutil.copy(src, src.with_name("old_clip_" + src.name))
rc, out = run("mark", "--config", str(c), "--asset", str(e))
check(rc == 1 and "musubi would TRAIN them" in out, "stale latent with no jsonl row")

root = T / "caption_drift"
c, ca, e = make_ws(root)
jl = root / "data_v10/dataset_v10_scene_f21.workspace.jsonl"
jl.write_text(jl.read_text().replace("Pudgy Penguins style", "Pudgy Penguin style", 1))
rc, out = run("mark", "--config", str(c), "--asset", str(e))
check(rc == 1 and "caption differs from the jsonl row" in out, "jsonl caption changed without re-caching")

root = T / "e5_rows"
c, ca, e = make_ws(root)
save_file({"E": torch.zeros(PH_N - 1, 32)}, str(e))
rc, out = run("mark", "--config", str(c), "--asset", str(e))
check(rc == 1 and "E5 has" in out, "E5 row count != phrase tokens")

root = T / "bad_dirname"
c, ca, e = make_ws(root)
c.write_text(c.read_text().replace("replay_f57", "replay57"))
(ca / "replay_f57").rename(ca / "replay57")
rc, out = run("mark", "--config", str(c), "--asset", str(e))
check(rc == 1 and "must end in scene_f<n> or replay_f<n>" in out, "cache dir not named scene_f<n>/replay_f<n>")

root = T / "cache_root"
c, ca, e = make_ws(root)
rc, out = run("check", "--config", str(c), "--cache-root", str(root / "elsewhere"))
check(rc == 1 and "is not under" in out, "--cache-root guards against a toml pointing at other caches")

common.finish()
