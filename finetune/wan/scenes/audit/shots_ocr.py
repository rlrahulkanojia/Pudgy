"""Per skit: scene-cut shot list (PySceneDetect AdaptiveDetector) + OCR on every
STEP-th frame. Writes <out>/<iteration>__<stem>.json. Usage: shots_ocr.py OUT probe.json"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
STEP = 6  # 0.25 s at 24 fps


def run(row):
    import cv2
    from rapidocr_onnxruntime import RapidOCR
    from scenedetect import detect, AdaptiveDetector

    f = Path(row["file"])
    dst = OUT / f"{row['iteration']}__{f.stem}.json"
    if dst.exists():
        return dst.name
    scenes = detect(str(f), AdaptiveDetector())
    shots = [(s.get_frames(), e.get_frames()) for s, e in scenes] or [(0, row["nb_read_frames"])]
    ocr = RapidOCR(det_limit_side_len=1920, det_limit_type="max", intra_op_num_threads=2)
    cap = cv2.VideoCapture(str(f))
    frames, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % STEP == 0:
            h, w = img.shape[:2]
            res, _ = ocr(img)
            dets = []
            for box, text, score in res or []:
                xs, ys = [p[0] for p in box], [p[1] for p in box]
                dets.append({"text": text, "score": round(float(score), 3),
                             "box": [round(min(xs) / w, 3), round(min(ys) / h, 3),
                                     round(max(xs) / w, 3), round(max(ys) / h, 3)]})
            frames.append({"frame": i, "dets": dets})
        i += 1
    dst.write_text(json.dumps({**row, "decoded": i, "shots": shots, "ocr": frames}))
    return dst.name


if __name__ == "__main__":
    OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(sys.argv[2]).read_text())
    with Pool(40, initializer=lambda: globals().__setitem__("OUT", OUT)) as p:
        for n, name in enumerate(p.imap_unordered(run, rows), 1):
            print(n, name, flush=True)
