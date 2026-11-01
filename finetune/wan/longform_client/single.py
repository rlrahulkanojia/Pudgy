"""Comparison: a whole video as ONE generation (241 frames = 10.0 s at 24 fps), no chaining.
Same prompt, start frame, seed (take 1's), model and sampler settings as the chained video;
only the length differs (Wan's native length is 81 frames).

python single.py <id> [frames=241]     -> takes/<id>_single_t1.mp4
then: python assemble_client.py <id> single   -> final/<id>_single_1080x1920.mp4
"""
import sys

sys.path.insert(0, "/workspace/longform_client")
import produce as PR                     # noqa: E402

vid = sys.argv[1]
frames = int(sys.argv[2]) if len(sys.argv) > 2 else 241
assert (frames - 1) % 4 == 0, "Wan needs 4n+1 frames"
[job] = PR.jobs_for(1, PR.select([f"{vid}:single"]))
job["frames"] = frames
print(job["tag"], frames, "frames, seed", job["seed"], flush=True)
PR.gen_v8.run("v8merged", [job], PR.TAKES_DIR)
print("SINGLE DONE", job["tag"], flush=True)
