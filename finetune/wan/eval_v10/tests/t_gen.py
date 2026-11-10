"""Mac tests for the v10 eval package (no GPU). <W> = a scratch dir; <PY> = a venv with numpy, scipy, opencv,
pillow, torch, transformers, rapidocr_onnxruntime. First build a Mac-rooted copy of the eval set:
    cd finetune/wan/eval_v10 && PUDGY_DATA=<Data> <PY> build_eval_v10.py --out <W>/eval_mac --root <W>/eval_mac
    mkdir -p <W>/tests && cp tests/fake_gen_asset.py <W>/tests/
Then:
    <PY> tests/t_tier1.py <W> <repo> <PY>          # Tier-1 gate / early stop / summary exit codes on synthetic results
    PUDGY_DATA=<Data> V10_EVAL=<W>/eval_mac RUN_NAME=mac-gen <PY> tests/t_gen.py <W> <repo>/finetune/wan
                                                   # generation against a fake musubi (--interactive protocol,
                                                   # E5 args, --max-new / REMAINING, 848x480 vs 1024^2, FLF, G5 gate)
    PUDGY_DATA=<Data> V10_EVAL=<W>/eval_mac RUN_NAME=mac-selftest <PY> suite_v10.py selftest   # G2 on real shots
"""
import json, os, subprocess, sys
from pathlib import Path
SP = Path(sys.argv[1]); WAN = Path(sys.argv[2])
sys.path.insert(0, str(WAN / "eval_v10"))
import common_v10 as C
C.C8.PY = Path(sys.executable); C.C8.MUSUBI = SP / "fake_musubi"; C.C8.MUSUBI.mkdir(exist_ok=True)
C.GEN_ASSET = SP / "tests/fake_gen_asset.py"
C.missing_files = lambda n: []
C.E5 = C.GEN_ASSET                       # any existing file
import suite_v10 as SU, g4_v10 as G4, g5_v10 as G5, phase0_v10 as P0
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
# 1. --max-new slices, seed-major order, resume
r1 = SU.gen("v10@50", limit=3, max_new=4)
r2 = SU.gen("v10@50", limit=3, max_new=4)
r3 = SU.gen("v10@50", limit=3)
d = SU.cfg_out("v10@50")
made = sorted(p.name for p in d.glob("*.mp4"))
print("remaining after slices:", r1, r2, r3, "| clips:", len(made), made[:4])
cmd = json.loads((d / ".raw_v10_50/cmd.json").read_text())
vs = [i for i, x in enumerate(cmd) if x == "--video_size"]
print("video_size args:", [cmd[i + 1:i + 3] for i in vs], "| high LoRAs:", cmd[cmd.index("--lora_weight_high_noise") + 1:cmd.index("--lora_multiplier_high_noise")])
print("low LoRAs:", cmd[cmd.index("--lora_weight") + 1:cmd.index("--lora_multiplier")])
fm = C.load_json(d / "frame_mismatch.json", {})
print("frame mismatches:", fm)
log = (d / ".raw_v10_50/gen.log").read_text()
print("log: lines with --ei:", sum(" --ei " in l for l in log.splitlines()), "| spliced:", log.count("[asset] spliced"))
# 2. FLF job carries --ei (jobs with flf)
jobs = SU.load_jobs()
fl = [j for j in jobs if j["flf"]][:1]
SU.C.generate("v10@50", SU.expand(fl, (42,), flf=True, flf_only=True), d, size=C.SCENE_SIZE)
print("FLF clip:", sorted(p.name for p in d.glob("*_flf_*.mp4")), "frames", C.frame_count(next(d.glob("*_flf_*.mp4"))))
# 3. g4 core+compose at 1024, budget across suites
print("g4 remaining (max_new 5):", G4.gen("v8merged", max_new=5), "| then all:", G4.gen("v8merged"))
c4 = json.loads((G4.out_dir("core", "v8merged") / ".raw_v8merged/cmd.json").read_text())
print("g4 video_size:", c4[c4.index("--video_size") + 1:c4.index("--video_size") + 3], "| core clips", len(list(G4.out_dir("core", "v8merged").glob("*.mp4"))),
      "compose clips", len(list(G4.out_dir("compose", "v8merged").glob("*.mp4"))))
# 4. g5: 24 clips + ID1 gate on static phone keyframes
print("g5 remaining (max_new 10):", G5.gen("v10@50", max_new=10), "| then:", G5.gen("v10@50"))
print("g5 gate exit:", G5.gate("v10@50"))
# 5. phase0 p1 with a budget spread over steps
print("p1 remaining (max_new 7):", P0.p1(7))
