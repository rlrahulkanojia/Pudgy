#!/usr/bin/env python3
"""Training set for the v9c2 pivotal LoRA stage (RESEARCH_v9c_fixes section 6, step 3).

Phone clips (21-41 frames) + replay upweighted to ~35% + v8 'sitting' anchors (>= 1 per 4 phone-sitting
samples), each group in its OWN cache directory: musubi trains on every latent in a cache dir, whatever
the jsonl lists. Cache files are symlinks into latents_v9c2_emb (span-annotated; replay spans are [0, 0],
which is what --preserve keys on).

    python build_pivotal_set_v9.py            # writes /workspace/data_v9c2/dataset_pivotal.toml
"""
import json
import os
from pathlib import Path

W = Path("/workspace")
EMB = W / "wan_cache/latents_v9c2_emb"
DATA = W / "data_v9c2"
MAX_F = 41
REPEATS = {"props": {}, "replay_sit": {"*": 7}, "replay_other": {"f21": 2, "*": 1}}


def group(name):
    if "phone_texting" in name:
        return "props"
    return "replay_sit" if "_sitting_" in name else "replay_other"


def main():
    caps = {}
    for jf in DATA.glob("dataset_v9c_phone_f??.workspace.jsonl"):
        for line in jf.read_text().splitlines():
            j = json.loads(line)
            caps[Path(j["video_path"]).stem] = j
    blocks, counts = [], {}
    for b in sorted(EMB.glob("f*"), key=lambda p: int(p.name[1:])):
        if int(b.name[1:]) > MAX_F:
            continue
        members = {}
        for te in b.glob("*_wan_te.safetensors"):
            stem = te.name[: -len("_wan_te.safetensors")]
            members.setdefault(group(stem), []).append(stem)
        for g, stems in sorted(members.items()):
            cdir = W / f"wan_cache/latents_v9c2_pivotal_{g}" / b.name
            cdir.mkdir(parents=True, exist_ok=True)
            for stem in stems:
                for f in list(b.glob(f"{stem}_wan_te.safetensors")) + list(b.glob(f"{stem}_*_wan.safetensors")):
                    dst = cdir / f.name
                    if not dst.exists():
                        dst.symlink_to(os.path.realpath(f))
            jl = DATA / f"pivotal_{g}_{b.name}.workspace.jsonl"
            jl.write_text("\n".join(json.dumps(caps[s], ensure_ascii=False) for s in sorted(stems)) + "\n")
            rep = REPEATS[g].get(b.name, REPEATS[g].get("*", 1))
            counts[g] = counts.get(g, 0) + len(stems) * rep
            blocks.append(f'[[datasets]]\nvideo_jsonl_file = "{jl}"\ncache_directory = "{cdir}"\n'
                          f'target_frames = [{b.name[1:]}]\nframe_extraction = "head"\nnum_repeats = {rep}\n'
                          f'# {g}: {len(stems)} clips x {rep}\n')
    tot = sum(counts.values())
    rp = counts.get("replay_sit", 0) + counts.get("replay_other", 0)
    hdr = ("# v9c2 pivotal LoRA stage (build_pivotal_set_v9.py): phone + replay (~35%) + sitting anchors,\n"
           f"# one cache dir per group. Samples per epoch: {counts} -> replay {rp}/{tot} = {rp / tot:.1%}\n"
           "[general]\nresolution = [1024, 1024]\nbatch_size = 1\nenable_bucket = true\nbucket_no_upscale = true\n\n")
    (DATA / "dataset_pivotal.toml").write_text(hdr + "\n".join(blocks))
    phone_sit = sum(1 for s in caps if "phone_texting_sitting" in s and int(s.rsplit("_f", 1)[1]) <= MAX_F)
    print(f"samples per epoch {counts}; replay {rp}/{tot} = {rp / tot:.1%}; "
          f"sitting anchors {counts.get('replay_sit', 0)} vs phone-sitting {phone_sit} (need >= {phone_sit / 4:.0f})")


if __name__ == "__main__":
    main()
