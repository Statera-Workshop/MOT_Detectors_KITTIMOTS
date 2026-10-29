#!/usr/bin/env python3
"""
MOT stage (2026-10-23): run a trained image detector frame by frame over video sequences, save the detections and the
per-frame detector latency. Trackers then run on the saved detections (tools/mot/run_tracker.py), so every tracker
sees exactly the same detections.

    # KITTI-MOTS val (wide input; checkpoint from the study dev3_mot_det_wide)
    UNIFIED_IMAGE_DATASET=kittimots UNIFIED_IMAGE_H=384 python tools/mot/dump_dets.py --source kittimots \
        --version x --img-size 1216 --weights ~/Physical_AI/projects/dev3_mot_det_wide/runs/image_x/checkpoints/...pth
    # FL36-MOT (pipeline check with an FL36 36-class checkpoint, square 608)
    UNIFIED_IMAGE_DATASET=fl36 python tools/mot/dump_dets.py --source fl36mot --version x --weights <fl36 ckpt>

Output: ~/mot_logs/dets/<source>/<version>.npz - per sequence "<seq>" rows [frame, x1, y1, x2, y2, score, class]
        (original image pixels), "<seq>__ms" detector ms per frame, "<seq>__pre_ms" preprocessing ms, "meta" json.
Latency protocol (progress tab 1): batch 1, fp32, model.eval, torch.no_grad, 50 warm-up forwards on the first frame,
then per frame torch.cuda.synchronize() -> forward + post-processing (confidence, NMS, letterbox inverse) ->
synchronize. Image decoding and letterboxing are timed separately (pre_ms), not included in ms.
[Unified] stored detections: score >= 0.05 after NMS (ByteTrack's low threshold is 0.1), profile NMS / max_det.
Stop the training queue while this runs if the latency numbers are going to be reported.
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config as C  # noqa: E402
from common.datasets.voc import CANVAS_H, letterbox, to_tensor  # noqa: E402
from common.mot.evaluate import load_fl36mot  # noqa: E402
from common.utils import load_checkpoint  # noqa: E402
from overfit_check import MODULE  # noqa: E402
from tasks.image_task import ImageTask  # noqa: E402


def frame_lists(source, split):
    if source == "kittimots":
        root = C.DATASETS_ROOT / "KITTI_MOTS_VOC"
        out = {}
        for line in filter(None, (root / "ImageSets" / "MOT" / f"{split}_seqmap.txt").read_text().split("\n")):
            seq, n = line.split()
            out[seq] = [str(root / "JPEGImages" / f"{seq}_{f:06d}.jpg") for f in range(int(n))]
        return out
    _, frames = load_fl36mot(C.DATASETS_ROOT / "FL36_MOT", split)
    return frames


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["kittimots", "fl36mot"], required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--version", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img-size", type=int, default=C.TASKS["image"]["img_size"])
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = torch.device("cuda")
    task = ImageTask()
    model = importlib.import_module(f"detectors.{MODULE[a.version]}.build").build_model("image", len(task.class_names))
    ckpt = load_checkpoint(Path(a.weights).expanduser(), model.to(device), device)
    model.eval()
    frames = frame_lists(a.source, a.split)
    out = Path(a.out or f"~/mot_logs/dets/{a.source}/{a.version}.npz").expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas = f"{a.img_size}x{CANVAS_H or a.img_size}"
    print(f"{a.version}: {sum(len(v) for v in frames.values())} frames in {len(frames)} sequences, input {canvas}, "
          f"classes {len(task.class_names)}, GPU {torch.cuda.get_device_name(0)}")

    first = Image.open(next(iter(frames.values()))[0]).convert("RGB")
    x = to_tensor(letterbox(first, a.img_size)[0]).unsqueeze(0).to(device)
    for _ in range(a.warmup):
        task.postprocess(model(x), [first.size], a.img_size, a.conf)
    torch.cuda.synchronize()

    arrays, all_ms = {}, []
    for seq, paths in frames.items():
        rows, ms, pre = [], [], []
        for f, p in enumerate(paths):
            t0 = time.perf_counter()
            im = Image.open(p).convert("RGB")
            x = to_tensor(letterbox(im, a.img_size)[0]).unsqueeze(0).to(device)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            det = task.postprocess(model(x), [im.size], a.img_size, a.conf)[0]
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            pre.append((t1 - t0) * 1000)
            ms.append((t2 - t1) * 1000)
            n = len(det["scores"])
            if n:
                rows.append(np.c_[np.full(n, f), det["boxes"], det["scores"], det["labels"]])
        arrays[seq] = np.concatenate(rows).astype(np.float32) if rows else np.zeros((0, 7), np.float32)
        arrays[f"{seq}__ms"] = np.array(ms, np.float32)
        arrays[f"{seq}__pre_ms"] = np.array(pre, np.float32)
        all_ms += ms
        print(f"  {seq}: {len(paths)} frames, {len(arrays[seq])} dets, det {np.median(ms):.1f} ms", flush=True)
    meta = {"version": a.version, "weights": str(a.weights), "source": a.source, "split": a.split,
            "input": canvas, "conf": a.conf, "classes": task.class_names, "epoch": ckpt.get("epoch"),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "det_ms_median": float(np.median(all_ms)), "det_ms_p95": float(np.percentile(all_ms, 95)),
            "params_M": sum(p.numel() for p in model.parameters()) / 1e6}
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)
    print(f"-> {out}  det {meta['det_ms_median']:.1f} ms median, {meta['det_ms_p95']:.1f} ms p95")


if __name__ == "__main__":
    main()
