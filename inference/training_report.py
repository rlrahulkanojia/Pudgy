#!/usr/bin/env python
"""
Generate a training-run report for the Pudgy CogVideoX1.5-5B-I2V LoRA.

It assembles, from the training log + dataset + the final checkpoint's generated video:
  * number of epochs / steps and the loss curve summary,
  * the exact command used (from report/run_command.txt captured at launch),
  * dataset details (clip count, resolution, fps, frames, duration, sample captions),
  * a per-frame consistency analysis of the generated video (adjacent-frame SSIM,
    color-histogram correlation, mean abs diff, and drift vs. the conditioning frame),
  * a contact-sheet montage of every frame, the raw frames, and the video stored locally,
  * a Markdown report tying it together.

Typical use (after training finishes and a video has been generated with
inference/eval_pudgy_lora.py into report/):

    python inference/training_report.py \
        --log /path/to/train.log \
        --output_root finetune/output_dir/pudgy-lora-v1 \
        --video finetune/output_dir/pudgy-lora-v1/report/final.mp4

Only the frame-analysis + report parts need no GPU; video generation is done separately
by eval_pudgy_lora.py so this script stays light and re-runnable.
"""

import argparse
import glob
import json
import os
import re

import cv2
import numpy as np


# ---------------------------------------------------------------- log parsing
def parse_training_log(log_path):
    """Pull run stats out of the trainer's stdout log."""
    info = {"log_path": log_path}
    if not log_path or not os.path.isfile(log_path):
        return info
    with open(log_path, errors="replace") as f:
        raw = f.read()
    text = raw.replace("\r", "\n")  # split the tqdm carriage-return progress bar

    def grab(pat, cast=str, default=None):
        m = re.search(pat, text)
        return cast(m.group(1)) if m else default

    info["num_examples"] = grab(r"Num examples = (\d+)", int)
    info["num_epochs"] = grab(r"Num epochs = (\d+)", int)
    info["batches_per_epoch"] = grab(r"Num batches each epoch = (\d+)", int)
    info["batch_size_per_device"] = grab(r"Instantaneous batch size per device = (\d+)", int)
    info["total_batch_size"] = grab(r"Total train batch size.*= (\d+)", int)
    info["grad_accum"] = grab(r"Gradient accumulation steps = (\d+)", int)
    info["total_steps"] = grab(r"Total optimization steps = (\d+)", int)

    # All (step, loss, lr) progress points. tqdm renders the postfix INSIDE the
    # bracket: "  17/2500 [03:57<9:44:48, 14.13s/it, loss=0.0247, lr=2.55e-6]".
    pts = re.findall(r"(\d+)/\d+ \[[^\]]*?loss=([0-9.eE+-]+),\s*lr=([0-9.eE+-]+)", text)
    losses, steps, lrs = [], [], []
    for s, l, lr in pts:
        try:
            steps.append(int(s)); losses.append(float(l)); lrs.append(float(lr))
        except ValueError:
            pass
    if losses:
        info["steps_reached"] = max(steps)
        info["loss_first"] = losses[0]
        info["loss_last"] = losses[-1]
        info["loss_min"] = min(losses)
        info["loss_mean_last50"] = float(np.mean(losses[-50:]))
        info["lr_last"] = lrs[-1]
        info["loss_series"] = list(zip(steps, losses))  # for a sparkline in the report

    # Elapsed wall time from the last tqdm "[MM:SS<..." / "[H:MM:SS<...".
    elapsed = re.findall(r"\[(\d+:\d+(?::\d+)?)<", text)
    info["elapsed_last"] = elapsed[-1] if elapsed else None
    info["it_per_step"] = grab(r"([0-9.]+)s/it", float)
    info["completed"] = "end_training" in text or "Steps: 100%" in text or \
        (info.get("steps_reached") and info.get("total_steps") and
         info["steps_reached"] >= info["total_steps"])
    return info


# ------------------------------------------------------------- dataset details
def dataset_details(dataset_dir):
    d = {"dataset_dir": dataset_dir}
    meta_path = os.path.join(dataset_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return d
    with open(meta_path) as f:
        meta = json.load(f)
    entries = meta if isinstance(meta, list) else meta.get("data", [])
    d["num_clips"] = len(entries)
    d["sample_captions"] = [e.get("text", "") for e in entries[:3]]
    # Probe the first clip with ffprobe.
    if entries:
        clip = os.path.join(dataset_dir, entries[0]["file_path"])
        try:
            import subprocess
            out = subprocess.check_output(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", clip])
            s = [x for x in json.loads(out)["streams"] if x["codec_type"] == "video"][0]
            num, den = (s.get("r_frame_rate", "16/1").split("/") + ["1"])[:2]
            d["clip_probe"] = {
                "example": entries[0]["file_path"],
                "width": s.get("width"), "height": s.get("height"),
                "nb_frames": s.get("nb_frames"),
                "fps": round(float(num) / float(den), 3) if float(den) else None,
                "codec": s.get("codec_name"),
            }
        except Exception as e:
            d["clip_probe_error"] = str(e)
    return d


# ------------------------------------------------- frame extraction + metrics
def read_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _ssim(a, b):
    """Grayscale SSIM (Gaussian window), no skimage dependency."""
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu_a = cv2.GaussianBlur(a, k, 1.5); mu_b = cv2.GaussianBlur(b, k, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.GaussianBlur(a * a, k, 1.5) - mu_a2
    sb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b2
    sab = cv2.GaussianBlur(a * b, k, 1.5) - mu_ab
    ssim_map = ((2 * mu_ab + C1) * (2 * sab + C2)) / ((mu_a2 + mu_b2 + C1) * (sa + sb + C2))
    return float(ssim_map.mean())


def _hist_corr(a, b):
    """Color-histogram correlation (palette stability), HSV, 0..1."""
    ha = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_RGB2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hb = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_RGB2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(ha, ha); cv2.normalize(hb, hb)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


def consistency_metrics(frames):
    """Per-adjacent-pair and vs-frame-0 metrics + summary flags."""
    n = len(frames)
    out = {"num_frames": n, "adjacent": [], "vs_frame0": []}
    if n < 2:
        return out
    for i in range(1, n):
        out["adjacent"].append({
            "pair": f"{i-1}->{i}",
            "ssim": round(_ssim(frames[i-1], frames[i]), 4),
            "hist_corr": round(_hist_corr(frames[i-1], frames[i]), 4),
            "mean_abs_diff": round(float(np.abs(frames[i].astype(int) - frames[i-1].astype(int)).mean()), 3),
        })
    for i in range(n):
        out["vs_frame0"].append({
            "frame": i,
            "ssim": round(_ssim(frames[0], frames[i]), 4),
            "hist_corr": round(_hist_corr(frames[0], frames[i]), 4),
        })
    adj_ssim = [a["ssim"] for a in out["adjacent"]]
    adj_hist = [a["hist_corr"] for a in out["adjacent"]]
    drift_hist = [v["hist_corr"] for v in out["vs_frame0"]]
    out["summary"] = {
        "adjacent_ssim_mean": round(float(np.mean(adj_ssim)), 4),
        "adjacent_ssim_min": round(float(np.min(adj_ssim)), 4),
        "adjacent_ssim_std": round(float(np.std(adj_ssim)), 4),
        "adjacent_hist_corr_mean": round(float(np.mean(adj_hist)), 4),
        "palette_drift_frame0_to_last": round(drift_hist[0] - drift_hist[-1], 4),
        # Heuristic flags — a sharp adjacent-SSIM dip suggests a flicker/cut/morph.
        "worst_adjacent_pair": out["adjacent"][int(np.argmin(adj_ssim))]["pair"],
        "flicker_suspect_pairs": [a["pair"] for a in out["adjacent"]
                                  if a["ssim"] < np.mean(adj_ssim) - 2 * (np.std(adj_ssim) or 1e-9)],
    }
    return out


def build_montage(frames, out_path, cols=6, thumb_w=180):
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    tw, th = thumb_w, int(thumb_w * h / w)
    rows = (len(frames) + cols - 1) // cols
    sheet = np.full((rows * (th + 24), cols * tw, 3), 30, np.uint8)
    for i, fr in enumerate(frames):
        t = cv2.resize(fr, (tw, th))
        r, c = divmod(i, cols)
        y, x = r * (th + 24), c * tw
        sheet[y + 24:y + 24 + th, x:x + tw] = t
        cv2.putText(sheet, f"f{i}", (x + 3, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1)
    cv2.imwrite(out_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return out_path


def _sparkline(series):
    if not series:
        return "(no loss points parsed)"
    vals = [l for _, l in series]
    lo, hi = min(vals), max(vals)
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[min(7, int((v - lo) / (hi - lo + 1e-12) * 7))] for _, v in series[::max(1, len(series)//60)])


# --------------------------------------------------------------------- report
def write_report(md_path, log_info, ds, metrics, montage_rel, video_rel, cmd_text, ckpt_label):
    L = []
    L.append(f"# Training run report — Pudgy CogVideoX1.5-5B-I2V LoRA\n")
    status = "✅ completed" if log_info.get("completed") else "⏳ in progress / partial"
    L.append(f"**Status:** {status}  ·  **Evaluated checkpoint:** `{ckpt_label}`\n")

    L.append("## 1. Training schedule\n")
    L.append(f"| Metric | Value |\n|---|---|")
    L.append(f"| Epochs (planned) | {log_info.get('num_epochs')} |")
    L.append(f"| Optimizer steps (planned) | {log_info.get('total_steps')} |")
    L.append(f"| Steps reached (from log) | {log_info.get('steps_reached')} |")
    L.append(f"| Examples / batches per epoch | {log_info.get('num_examples')} / {log_info.get('batches_per_epoch')} |")
    L.append(f"| Batch size/device · grad-accum · effective | {log_info.get('batch_size_per_device')} · {log_info.get('grad_accum')} · {log_info.get('total_batch_size')} |")
    L.append(f"| Wall time (last log tick) | {log_info.get('elapsed_last')} |")
    L.append(f"| Loss first → last | {log_info.get('loss_first')} → {log_info.get('loss_last')} |")
    L.append(f"| Loss min · mean(last 50) | {log_info.get('loss_min')} · {log_info.get('loss_mean_last50')} |")
    L.append(f"| Final LR | {log_info.get('lr_last')} |\n")
    L.append(f"Loss trend: `{_sparkline(log_info.get('loss_series', []))}`\n")

    L.append("## 2. Command used\n")
    L.append("```bash\n" + cmd_text.strip() + "\n```\n")

    L.append("## 3. Dataset\n")
    L.append(f"- **Directory:** `{ds.get('dataset_dir')}`")
    L.append(f"- **Clips:** {ds.get('num_clips')}")
    p = ds.get("clip_probe")
    if p:
        L.append(f"- **Per-clip:** {p['width']}×{p['height']} · {p['nb_frames']} frames · {p['fps']} fps · {p['codec']} (e.g. `{p['example']}`)")
    if ds.get("sample_captions"):
        L.append("- **Sample captions:**")
        for c in ds["sample_captions"]:
            L.append(f"  - {c[:200]}")
    L.append("")

    L.append("## 4. Generated video — consistency analysis\n")
    if video_rel:
        L.append(f"- **Video (stored locally):** `{video_rel}`")
    if montage_rel:
        L.append(f"- **All-frames contact sheet:** `{montage_rel}`\n")
        L.append(f"![frames]({os.path.basename(montage_rel)})\n")
    s = metrics.get("summary")
    if s:
        L.append(f"| Consistency metric | Value | Reading |\n|---|---|---|")
        L.append(f"| Adjacent-frame SSIM (mean) | {s['adjacent_ssim_mean']} | 1.0 = identical; >~0.85 = smooth temporal coherence |")
        L.append(f"| Adjacent-frame SSIM (min) | {s['adjacent_ssim_min']} | worst transition, at pair {s['worst_adjacent_pair']} |")
        L.append(f"| Adjacent-frame SSIM (std) | {s['adjacent_ssim_std']} | low = uniform motion; spikes = flicker |")
        L.append(f"| Adjacent hist-corr (mean) | {s['adjacent_hist_corr_mean']} | palette stability frame-to-frame |")
        L.append(f"| Palette drift f0→last | {s['palette_drift_frame0_to_last']} | ~0 = colors hold; large = identity/color drift |")
        L.append(f"| Flicker-suspect pairs | {', '.join(s['flicker_suspect_pairs']) or 'none'} | adjacent SSIM >2σ below mean |")
        L.append("")
    L.append("### Per-frame drift vs. conditioning frame (frame 0)\n")
    L.append("| frame | SSIM vs f0 | hist-corr vs f0 |\n|---|---|---|")
    for v in metrics.get("vs_frame0", []):
        L.append(f"| {v['frame']} | {v['ssim']} | {v['hist_corr']} |")
    L.append("")
    L.append("### Qualitative evaluation (fill in from visual inspection)\n")
    L.append("- **Character identity (Pax/Polly):** _<consistent shape/color/face across frames? any morphing?>_")
    L.append("- **Style fidelity (thick outlines, flat pastel):** _<held throughout?>_")
    L.append("- **Motion:** _<gentle bouncy per the caption, or artifacts/warping?>_")
    L.append("- **Temporal coherence:** _<flicker, background stability>_")
    L.append("- **Verdict:** _<usable / needs earlier checkpoint / overfit>_\n")

    with open(md_path, "w") as f:
        f.write("\n".join(L))
    return md_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="Path to the training stdout log")
    ap.add_argument("--output_root", default="finetune/output_dir/pudgy-lora-v1",
                    help="Training output dir (holds report/ and run_command.txt)")
    ap.add_argument("--dataset_dir", default="/workspace/training_dataset")
    ap.add_argument("--video", default=None, help="Generated video to analyze (default: report/final.mp4)")
    ap.add_argument("--checkpoint_label", default="final", help="Label of the checkpoint evaluated")
    args = ap.parse_args()

    report_dir = os.path.join(args.output_root, "report")
    os.makedirs(report_dir, exist_ok=True)
    frames_dir = os.path.join(report_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    log_info = parse_training_log(args.log)
    ds = dataset_details(args.dataset_dir)

    cmd_path = os.path.join(report_dir, "run_command.txt")
    cmd_text = open(cmd_path).read() if os.path.isfile(cmd_path) else "(run_command.txt not found)"

    video = args.video or os.path.join(report_dir, "final.mp4")
    metrics, montage_rel, video_rel = {"num_frames": 0}, None, None
    if os.path.isfile(video):
        frames = read_frames(video)
        for i, fr in enumerate(frames):
            cv2.imwrite(os.path.join(frames_dir, f"frame_{i:03d}.png"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        metrics = consistency_metrics(frames)
        montage_rel = build_montage(frames, os.path.join(report_dir, "frames_montage.png"))
        montage_rel = os.path.relpath(montage_rel, report_dir) if montage_rel else None
        video_rel = os.path.relpath(video, report_dir)
        print(f"Analyzed {len(frames)} frames from {video}")
    else:
        print(f"WARN: no video at {video} — report will omit the video analysis. "
              f"Generate one first with inference/eval_pudgy_lora.py --output_path {video}")

    with open(os.path.join(report_dir, "metrics.json"), "w") as f:
        json.dump({"log": {k: v for k, v in log_info.items() if k != "loss_series"},
                   "dataset": ds, "consistency": metrics}, f, indent=2)

    md = write_report(os.path.join(report_dir, "REPORT.md"), log_info, ds, metrics,
                      montage_rel, video_rel, cmd_text, args.checkpoint_label)
    print(f"Report written: {md}")


if __name__ == "__main__":
    main()
