#!/usr/bin/env python3
"""v10: mark musubi's text-encoder caches for the preservation loss and the E5 phone splice
(Training_Approach_v10.md section 0, Q10 and Q17; section 4 "preservation" and "phone").

Run on the GPU box AFTER musubi's latent + T5 caching (and verify_cache_v8.py), BEFORE training,
with the same python and HF cache as the T5 caching step (`mark` needs the umT5 tokenizer):

  python finetune/wan/cache_mark_v10.py mark  --config /workspace/data_v10/dataset_config_v10.workspace.toml \
      --asset /workspace/wan_output/e5/pudgy-v9c2-asset-dual-step00000150.asset.safetensors
  python finetune/wan/cache_mark_v10.py check --config <same> --asset <same> [--cache-root /workspace/wan_cache/latents_v10]
  python finetune/wan/cache_mark_v10.py smoke --config <same> --asset <same> --out /workspace/wan_cache/latents_v10_smoke

`mark` adds two tensors to EVERY `<stem>_wan_te.safetensors` of every [[datasets]] block:
  varlen_preserve   int64 [1]  1 in replay_f<n> cache dirs, 0 in scene_f<n> dirs  -> batch["preserve"]
  varlen_assetspan  int64 [2]  token span [s, e) of the registry phone phrase, [0, 0] if absent -> batch["assetspan"]
and metadata pudgy_mark=v10, pudgy_kind, pudgy_span, pudgy_caption_sha (the caption it was computed
from), pudgy_phrase_sha / pudgy_phrase_tokens (the registry phrase it looked for). musubi v0.3.4's
BucketBatchManager.__getitem__ (dataset/bucket.py) loads latent + _te file per sample, strips the
`varlen_` prefix and passes each varlen key to the trainer as a per-sample list, so
`asset_embed_v9.py train --preserve-marker key` reads batch["preserve"] and batch["assetspan"]
(E5 is added to rows [s, e) of the cached T5 output, exactly as at inference).

Replay is decided by the cache dir (replay_f<n>), never by the absence of a phone span: in v10 most
SCENE windows do not name the phone either (671 of 745), and must get plain data MSE.

IN PLACE, not a mirror dir. musubi trains on every `*_wan.safetensors` latent in a cache_directory
and pairs it with `<stem>_wan_te.safetensors` in the same dir (image_video_dataset.py,
VideoDataset.prepare_for_training; a latent without a _te file is skipped with only a warning). A
mirror (v9c2's `asset_embed_v9.py cache`) needs a second toml and inherits or accumulates stale
latents silently. In place keeps one toml and one set of files, and this script checks the file sets:
every latent has a jsonl row (else musubi would train a stale clip), every row has exactly one latent
with the block's frame count and a _te file (else musubi skips it), every _te file has a row.
Writes are atomic (temp file without the _te suffix, then os.replace), so musubi's cache globbing and
its post-caching cleanup never see a half-written file. If T5 caching is re-run later, musubi merges
into the existing _te file (cache_io.save_text_encoder_output_cache_common, merge_existing=True), so
the marks survive; if the caption changed, `check` sees pudgy_caption_sha != the new caption and fails
(re-run `mark`). `check` is what train_pudgy_v10.sh runs before every launch.

Inconsistencies (exit 1): a replay caption naming the phone (phrase or "iFin"), a missing/stale/extra
cache file, a _te caption that differs from its jsonl row, T5 rows != token count, the phrase in the
text but no token span (or twice), a span length != E5's rows, a block whose cache dir is not
scene_f<n> / replay_f<n>, two blocks sharing a cache dir, no replay sample at all.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import asset_embed_v9 as AE  # noqa: E402  (registry phrase, tokenizer, span finder: the v9c2 V0-validated code)

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

ARCH = "wan"
MARK = "v10"
PRESERVE_KEY = AE.PRESERVE_KEY
SPAN_KEY = AE.SPAN_KEY
LAT_SUFFIX = f"_{ARCH}.safetensors"
TE_SUFFIX = f"_{ARCH}_te.safetensors"
SMOKE_TAG = ".cache_mark_v10_smoke"


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Report:
    def __init__(self):
        self.errors, self.warnings = [], []

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def dump(self, limit=25):
        for w in self.warnings[:limit]:
            print(f"  warning: {w}")
        if len(self.warnings) > limit:
            print(f"  ... {len(self.warnings) - limit} more warnings")
        for e in self.errors[:limit]:
            print(f"  ERROR: {e}")
        if len(self.errors) > limit:
            print(f"  ... {len(self.errors) - limit} more errors")


def load_blocks(config, rep, cache_root=None):
    """[[datasets]] blocks of a musubi toml with their kind (scene / replay) and jsonl rows."""
    cfg = tomllib.loads(Path(config).read_text())
    blocks, seen = [], {}
    for i, ds in enumerate(cfg.get("datasets", [])):
        cache = Path(ds.get("cache_directory", ""))
        m = re.fullmatch(r"(scene|replay)_f(\d+)", cache.name)
        jl = ds.get("video_jsonl_file") or ds.get("jsonl_file")
        tag = f"block {i} ({cache.name or '?'})"
        if not m:
            rep.err(f"{tag}: cache_directory must end in scene_f<n> or replay_f<n>: {cache}")
            continue
        if not jl:
            rep.err(f"{tag}: no video_jsonl_file")
            continue
        kind, n = m.group(1), int(m.group(2))
        other = "scene" if kind == "replay" else "replay"
        if f"_{other}_" in Path(jl).name:
            rep.err(f"{tag}: jsonl {Path(jl).name} is a {other} file but the cache dir says {kind}")
        if ds.get("target_frames") not in (None, [n]):
            rep.err(f"{tag}: target_frames {ds.get('target_frames')} but the cache dir says f{n}")
        key = os.path.realpath(cache)
        if key in seen:
            rep.err(f"{tag}: shares its cache dir with block {seen[key]} (musubi would train its latents twice)")
        seen[key] = i
        if cache_root and not key.startswith(os.path.realpath(cache_root) + os.sep):
            rep.err(f"{tag}: cache dir {cache} is not under {cache_root}")
        rows, dup = {}, 0
        try:
            for line in Path(jl).read_text().splitlines():
                if line.strip():
                    j = json.loads(line)
                    stem = Path(j["video_path"]).stem
                    dup += stem in rows
                    rows[stem] = j["caption"]
        except (OSError, ValueError, KeyError) as e:
            rep.err(f"{tag}: cannot read {jl}: {e}")
            continue
        if dup:
            rep.err(f"{tag}: {dup} duplicate video stems in {jl}")
        blocks.append({"i": i, "ds": ds, "cache": cache, "kind": kind, "frames": n, "jsonl": jl, "rows": rows})
    if not blocks:
        rep.err(f"no usable [[datasets]] blocks in {config}")
    if blocks and not any(b["kind"] == "replay" for b in blocks):
        rep.err("no replay_f<n> block: nothing to preserve on")
    return cfg, blocks


def file_sets(b, rep):
    """stem -> (latent path, _te path) for the rows of block b, checking musubi's pairing rules."""
    cache, tag = b["cache"], b["cache"].name
    if not cache.is_dir():
        rep.err(f"{tag}: cache dir missing: {cache}")
        return {}
    lat = {}
    for p in sorted(cache.glob(f"*{LAT_SUFFIX}")):
        toks = p.name[: -len(LAT_SUFFIX)].split("_")         # <stem>_<pos>-<count>_<W>x<H>, as musubi parses it
        try:
            count = int(toks[-2].split("-")[1])
            w, h = (int(x) for x in toks[-1].split("x"))
            stem = "_".join(toks[:-2])
        except (IndexError, ValueError):
            rep.err(f"{tag}: latent name musubi cannot parse: {p.name}")
            continue
        lat.setdefault(stem, []).append((p, count, w, h))
    te = {p.name[: -len(TE_SUFFIX)]: p for p in cache.glob(f"*{TE_SUFFIX}")}
    stale = sorted(set(lat) - set(b["rows"]))
    if stale:
        rep.err(f"{tag}: {len(stale)} latents with no jsonl row - musubi would TRAIN them (e.g. {stale[0]})")
    extra_te = sorted(set(te) - set(b["rows"]))
    if extra_te:
        rep.err(f"{tag}: {len(extra_te)} _te files with no jsonl row (e.g. {extra_te[0]})")
    out = {}
    for stem in sorted(b["rows"]):
        ls = lat.get(stem, [])
        if len(ls) != 1:
            rep.err(f"{tag}: {stem}: {len(ls)} latent files (want 1)")
            continue
        if ls[0][1] != b["frames"]:
            rep.err(f"{tag}: {stem}: latent has {ls[0][1]} frames, block is f{b['frames']}")
        if stem not in te:
            rep.err(f"{tag}: {stem}: _te file missing - musubi would skip this sample")
            continue
        out[stem] = {"latent": ls[0][0], "te": te[stem], "tokens": ((ls[0][1] - 1) // 4 + 1) * (ls[0][2] // 16) * (ls[0][3] // 16)}
    return out


def read_te(path):
    with safe_open(str(path), "pt") as f:
        md = dict(f.metadata() or {})
        keys = list(f.keys())
        t5 = [k for k in keys if k.startswith("varlen_t5_")]
        rows = f.get_slice(t5[0]).get_shape()[0] if len(t5) == 1 else None
        pres = f.get_tensor(PRESERVE_KEY).tolist() if PRESERVE_KEY in keys else None
        span = f.get_tensor(SPAN_KEY).tolist() if SPAN_KEY in keys else None
    return md, rows, pres, span, len(t5)


def names_phone(caption, ph):
    return ph in caption or re.search(r"ifin", caption, re.I) is not None


def verify(b, stem, f, ph, ph_tok, e_rows, rep):
    """Read-only check of one marked _te file. Returns (kind, span) or None on error."""
    tag = f"{b['cache'].name}/{stem}"
    cap = b["rows"][stem]
    try:
        md, rows, pres, span, n_t5 = read_te(f["te"])
    except Exception as e:  # noqa: BLE001 - any unreadable file is an inconsistency
        rep.err(f"{tag}: unreadable _te: {e.__class__.__name__}: {e}")
        return None
    bad = []
    if n_t5 != 1:
        bad.append(f"{n_t5} varlen_t5_* tensors")
    if md.get("caption1") != cap:
        bad.append("its caption differs from the jsonl row (stale T5 cache: re-run T5 caching, then mark)")
    if md.get("pudgy_mark") != MARK or pres is None or span is None:
        bad.append("not marked (run cache_mark_v10.py mark)")
    elif md.get("pudgy_caption_sha") != sha(md.get("caption1", "")) or md.get("pudgy_caption_sha") != sha(cap):
        bad.append("marked for another caption (re-run cache_mark_v10.py mark)")
    elif md.get("pudgy_phrase_sha") != sha(ph):
        bad.append("marked for another registry phrase (re-run cache_mark_v10.py mark)")
    if bad:
        rep.err(f"{tag}: " + "; ".join(bad))
        return None
    s, e = (int(x) for x in span)
    want = 1 if b["kind"] == "replay" else 0
    if pres != [want]:
        bad.append(f"varlen_preserve {pres}, want [{want}] in a {b['kind']} dir")
    if md.get("pudgy_kind") != b["kind"]:
        bad.append(f"marked as {md.get('pudgy_kind')}, lives in a {b['kind']} dir")
    if b["kind"] == "replay":
        if names_phone(cap, ph):
            bad.append("replay caption names the phone")
        if (s, e) != (0, 0):
            bad.append(f"replay sample with an asset span {[s, e]}")
    else:
        if (ph in cap) != (e > s):
            bad.append(f"phrase in caption: {ph in cap}, token span {[s, e]}")
        if e > s:
            if not (0 <= s < e <= (rows or 0)):
                bad.append(f"span {[s, e]} outside the {rows} T5 rows")
            if e - s != int(md.get("pudgy_phrase_tokens", -1)) or (ph_tok is not None and e - s != ph_tok):
                bad.append(f"span length {e - s} != phrase tokens {md.get('pudgy_phrase_tokens')}")
            if e_rows is not None and e - s != e_rows:
                bad.append(f"span length {e - s} != E5 rows {e_rows}")
        elif (s, e) != (0, 0):
            bad.append(f"empty span written as {[s, e]}")
    if bad:
        rep.err(f"{tag}: " + "; ".join(bad))
        return None
    return b["kind"], (s, e)


def e5_rows(path, rep):
    if not path:
        return None
    try:
        return int(load_file(path)["E"].shape[0])
    except Exception as e:  # noqa: BLE001
        rep.err(f"cannot read E from {path}: {e}")
        return None


def run_check(config, asset, cache_root, quiet=False):
    """All read-only checks. Returns (ok, per-stem info, blocks)."""
    rep = Report()
    _, blocks = load_blocks(config, rep, cache_root)
    ph = AE.phrase()
    e_rows = e5_rows(asset, rep)
    counts = {"replay": 0, "scene": 0, "phone": 0, "short_name_only": 0}
    info = {}
    for b in blocks:
        fs = file_sets(b, rep)
        nb = {"replay": 0, "scene": 0, "phone": 0}
        for stem, f in fs.items():
            r = verify(b, stem, f, ph, None, e_rows, rep)
            if r is None:
                continue
            kind, (s, e) = r
            nb[kind] += 1
            nb["phone"] += e > s
            if kind == "scene" and e <= s and names_phone(b["rows"][stem], ph):
                counts["short_name_only"] += 1
            info[(b["i"], stem)] = {**f, "kind": kind, "span": (s, e)}
        for k in nb:
            counts[k] += nb[k]
        if not quiet:
            print(f"  {b['cache'].name:<12} {len(b['rows']):5d} rows  {len(fs):5d} paired  "
                  f"{'replay-marked ' + str(nb['replay']) if b['kind'] == 'replay' else 'scene ' + str(nb['scene']) + ' (phone ' + str(nb['phone']) + ')'}")
    if counts["replay"] == 0:
        rep.err("no replay-marked _te file: preservation would never fire")
    if counts["short_name_only"]:
        rep.warn(f"{counts['short_name_only']} scene captions mention iFin without the full phrase: no E5 splice there")
    if counts["phone"] == 0:
        rep.warn("no scene window names the phone: E5 is never spliced")
    ok = not rep.errors
    rep.dump()
    print(f"CACHE_MARK_CHECK {'OK' if ok else 'FAIL'} replay={counts['replay']} scene={counts['scene']} "
          f"phone={counts['phone']} errors={len(rep.errors)} config={config}", flush=True)
    return ok, info, blocks


def cmd_mark(a):
    rep = Report()
    _, blocks = load_blocks(a.config, rep, a.cache_root)
    ph = AE.phrase()
    e_rows = e5_rows(a.asset, rep)
    tok = AE.tokenizer()
    ph_ids = AE.token_ids(tok, ph)[:-1]                      # drop </s>, as asset_embed_v9 does
    if e_rows is not None and e_rows != len(ph_ids):
        rep.err(f"E5 has {e_rows} rows, the registry phrase has {len(ph_ids)} tokens")
    plan = []
    for b in blocks:
        for stem, f in file_sets(b, rep).items():
            cap = b["rows"][stem]
            tag = f"{b['cache'].name}/{stem}"
            try:
                md, rows, pres, span, n_t5 = read_te(f["te"])
            except Exception as e:  # noqa: BLE001
                rep.err(f"{tag}: unreadable _te: {e.__class__.__name__}: {e}")
                continue
            if n_t5 != 1:
                rep.err(f"{tag}: {n_t5} varlen_t5_* tensors")
                continue
            if md.get("caption1") != cap:
                rep.err(f"{tag}: _te caption differs from the jsonl row (stale T5 cache: re-run T5 caching)")
                continue
            if f["te"].is_symlink():
                rep.err(f"{tag}: _te is a symlink; mark the source cache instead")
                continue
            ids = AE.token_ids(tok, cap)
            if rows != len(ids):
                rep.err(f"{tag}: {rows} T5 rows but the caption has {len(ids)} tokens")
                continue
            if b["kind"] == "replay":
                if names_phone(cap, ph):
                    rep.err(f"{tag}: replay caption names the phone")
                    continue
                s, e = 0, 0
            else:
                try:
                    s, e = AE.find_span(ids, ph_ids)
                except ValueError:
                    rep.err(f"{tag}: the phone phrase occurs more than once")
                    continue
                if (ph in cap) != (e > s):
                    rep.err(f"{tag}: phrase in caption {ph in cap} but token span {[s, e]}")
                    continue
            want_md = {"pudgy_mark": MARK, "pudgy_kind": b["kind"], "pudgy_span": f"{s},{e}",
                       "pudgy_caption_sha": sha(cap), "pudgy_phrase_sha": sha(ph),
                       "pudgy_phrase_tokens": str(len(ph_ids))}
            same = (pres == [int(b["kind"] == "replay")] and span == [s, e]
                    and all(md.get(k) == v for k, v in want_md.items()))
            plan.append((f["te"], b["kind"], s, e, want_md, same))
    if rep.errors:
        rep.dump()
        print(f"CACHE_MARK FAIL: {len(rep.errors)} inconsistencies, nothing written", flush=True)
        sys.exit(1)
    written = 0
    for te, kind, s, e, want_md, same in plan:
        if same:
            continue
        written += 1
        if a.dry_run:
            continue
        with safe_open(str(te), "pt") as f:
            md = dict(f.metadata() or {})
        sd = load_file(te)
        sd[PRESERVE_KEY] = torch.tensor([int(kind == "replay")], dtype=torch.int64)
        sd[SPAN_KEY] = torch.tensor([s, e], dtype=torch.int64)
        md.update(want_md)
        tmp = te.with_name(te.name + ".marktmp")              # not *_te.safetensors: musubi never globs it
        save_file(sd, str(tmp), metadata=md)
        os.replace(tmp, te)
    n_rep = sum(k == "replay" for _, k, *_ in plan)
    n_phone = sum(e > s for _, k, s, e, *_ in plan if k == "scene")
    print(f"CACHE_MARK {'DRY-RUN ' if a.dry_run else ''}{'would write' if a.dry_run else 'wrote'} {written}, "
          f"unchanged {len(plan) - written}  |  replay {n_rep}, scene {len(plan) - n_rep} (phone span {n_phone}), "
          f"phrase {len(ph_ids)} tokens", flush=True)
    if a.dry_run:
        sys.exit(0)
    ok, _, _ = run_check(a.config, a.asset, a.cache_root)
    sys.exit(0 if ok else 1)


def cmd_check(a):
    ok, _, _ = run_check(a.config, a.asset, a.cache_root)
    sys.exit(0 if ok else 1)


def toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {v!r} to toml")


def cmd_smoke(a):
    """Three samples, one batch each: the heaviest replay sample (preservation: base pass + training
    pass at the largest token count), the heaviest scene window naming the phone (E5 splice) and the
    heaviest scene window without it (plain data loss). Symlinks into the marked cache, so the smoke
    run reads exactly the files training will read."""
    ok, info, blocks = run_check(a.config, a.asset, a.cache_root, quiet=True)
    if not ok:
        print("SMOKE SET NOT BUILT: the source cache fails the check", flush=True)
        sys.exit(1)
    pick = {}
    for (bi, stem), f in info.items():
        role = "replay" if f["kind"] == "replay" else ("phone" if f["span"][1] > f["span"][0] else "plain")
        cur = pick.get(role)
        if cur is None or (f["tokens"], stem) > (cur[1]["tokens"], cur[2]):
            pick[role] = (bi, f, stem)
    missing = [r for r in ("replay", "phone", "plain") if r not in pick]
    if missing:
        print(f"SMOKE SET NOT BUILT: no {', '.join(missing)} sample in {a.config}", flush=True)
        sys.exit(1)
    out = Path(a.out)
    if out.exists():
        if not (out / SMOKE_TAG).exists():
            sys.exit(f"{out} exists and was not made by this script: refusing to delete it")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / SMOKE_TAG).write_text("built by cache_mark_v10.py smoke; safe to delete\n")
    cfg = tomllib.loads(Path(a.config).read_text())
    lines = [f"# smoke set from {a.config} (cache_mark_v10.py smoke)", "[general]"]
    lines += [f"{k} = {toml_value(v)}" for k, v in cfg.get("general", {}).items()]
    by_i = {b["i"]: b for b in blocks}
    for role in ("replay", "phone", "plain"):
        bi, f, stem = pick[role]
        b = by_i[bi]
        d = out / role / f"{b['kind']}_f{b['frames']}"     # leaf named like the source, so `check` reads it
        d.mkdir(parents=True)
        for src in (f["latent"], f["te"]):
            (d / src.name).symlink_to(os.path.realpath(src))
        jl = out / f"{role}.jsonl"
        row = {"video_path": next(json.loads(x)["video_path"] for x in Path(b["jsonl"]).read_text().splitlines()
                                  if x.strip() and Path(json.loads(x)["video_path"]).stem == stem),
               "caption": b["rows"][stem]}
        jl.write_text(json.dumps(row) + "\n")
        lines += ["", "[[datasets]]"]
        for k, v in b["ds"].items():
            if k in ("cache_directory", "video_jsonl_file", "jsonl_file"):
                continue
            lines.append(f"{k} = {toml_value(v)}")
        lines += [f"video_jsonl_file = {toml_value(str(jl))}", f"cache_directory = {toml_value(str(d))}"]
        print(f"  smoke {role:<6} {b['cache'].name}/{stem}  tokens {f['tokens']}  span {list(f['span'])}")
    (out / "smoke.toml").write_text("\n".join(lines) + "\n")
    print(f"SMOKE SET {out / 'smoke.toml'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("mark", "check", "smoke"):
        p = sub.add_parser(n)
        p.add_argument("--config", required=True, help="musubi dataset toml (the one training uses)")
        p.add_argument("--asset", help="E5 .asset.safetensors: phone spans must have its row count")
        p.add_argument("--cache-root", help="every cache_directory must be under this dir")
        if n == "mark":
            p.add_argument("--dry-run", action="store_true")
        if n == "smoke":
            p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"mark": cmd_mark, "check": cmd_check, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
