"""Format + integrity check for every skit video: codec, size, fps, duration,
declared vs decodable frames, decoder errors. Writes probe.json."""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

roots = [Path(p) for p in sys.argv[2:]]
out = Path(sys.argv[1])
files = sorted(f for r in roots for f in r.iterdir() if f.suffix.lower() in (".mp4", ".mov"))


def probe(f):
    j = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(f)],
        capture_output=True, text=True).stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    num, den = map(int, v["r_frame_rate"].split("/"))
    errs = subprocess.run(["ffmpeg", "-v", "error", "-i", str(f), "-map", "0:v:0", "-f", "null", "-"],
                          capture_output=True, text=True).stderr.strip().splitlines()
    return {
        "file": str(f), "iteration": f.parts[-3], "name": f.name,
        "codec": v["codec_name"], "pix_fmt": v.get("pix_fmt"),
        "w": v["width"], "h": v["height"], "fps": num / den,
        "duration": float(j["format"]["duration"]),
        "nb_frames": int(v.get("nb_frames", -1)), "nb_read_frames": int(v.get("nb_read_frames", -1)),
        "audio": any(s["codec_type"] == "audio" for s in j["streams"]),
        "decode_errors": len(errs), "first_error": errs[0] if errs else "",
    }


with ThreadPoolExecutor(32) as ex:
    rows = list(ex.map(probe, files))
out.write_text(json.dumps(rows, indent=1))
print(len(rows), "files probed")
