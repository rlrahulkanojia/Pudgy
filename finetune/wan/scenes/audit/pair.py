"""Pair every skit video with its storyboard PDF(s); report PDF text/page/image stats.
Usage: pair.py OUT.json RAW_DIR"""
import hashlib, json, re, sys
from pathlib import Path
import pymupdf

raw = Path(sys.argv[2])
ALIAS = {"arm": "numbarm", "bicept": "biceps", "comfy": "gfcomfy", "sleep": "moviesleep",
         "thinkingofher": "thinkingofherd", "fixer": "fixer2", "customizing": "customize9x16"}


def key(name):
    s = Path(name).stem.lower()
    s = re.sub(r"^(2d_pp_|pp[-_]|content pipeline |menco )", "", s)
    s = re.sub(r"(content pipeline)$", "", s)
    s = re.sub(r"([-_]base.*|_base_1)$", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return ALIAS.get(s, s)


videos = sorted(p for it in ("iteration_1/30_videos", "iteration_2/72_videos") for p in (raw / it).iterdir())
pdfs = sorted(p for it in ("iteration_1/30_prompts", "iteration_2/72_prompts") for p in (raw / it).iterdir())
info = {}
for p in pdfs:
    d = pymupdf.open(p)
    text = "".join(pg.get_text() for pg in d)
    info[str(p)] = {"key": key(p.name), "md5": hashlib.md5(p.read_bytes()).hexdigest(), "pages": len(d),
                    "chars": len(text), "images": sum(len(pg.get_images()) for pg in d)}
rows = []
for v in videos:
    k = key(v.name)
    if "animatic" in v.name.lower():
        rows.append({"video": str(v), "key": k, "kind": "animatic", "pdfs": []})
        continue
    m = [p for p, i in info.items() if i["key"] == k]
    rows.append({"video": str(v), "key": k, "kind": "skit", "pdfs": m,
                 "unique_pdfs": len({info[p]["md5"] for p in m})})
used = {p for r in rows for p in r["pdfs"]}
Path(sys.argv[1]).write_text(json.dumps({"rows": rows, "pdf": info, "unused": sorted(set(info) - used)}, indent=1))
sk = [r for r in rows if r["kind"] == "skit"]
print("skits", len(sk), "with storyboard", sum(bool(r["pdfs"]) for r in sk))
print("no storyboard:", [Path(r["video"]).name for r in sk if not r["pdfs"]])
print("storyboard w/o video:", [Path(p).name for p in sorted(set(info) - used)])
print("skits whose 2 PDFs differ:", [Path(r["video"]).name for r in sk if r.get("unique_pdfs", 0) > 1])
pg = [i["pages"] for i in info.values()]; ch = [i["chars"] for i in info.values()]
print(f"PDF pages {min(pg)}-{max(pg)}; text chars min {min(ch)} median {sorted(ch)[len(ch)//2]}; "
      f"PDFs with <200 chars: {sum(c < 200 for c in ch)}")
