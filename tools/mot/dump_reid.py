#!/usr/bin/env python3
"""
MOT stage (2026-10-23): add appearance features to a detections file from dump_dets.py.

    python tools/mot/dump_reid.py --dets ~/mot_logs/dets/kittimots/x.npz          # -> x_reid.npz next to it

For every detection with score >= --min-score (0.1 = the lowest score any tracker uses) the box is cropped from
the frame and embedded by the shared ReID model; other rows get a zero vector. Stored as "<seq>__feat" (float16,
rows aligned with "<seq>"). ReID latency per frame ("<seq>__reid_ms"): crops of the frame as one batch, from the
decoded frame to the normalised features, torch.cuda.synchronize() on both sides, after 20 warm-up batches. As for
the detector, the GPU must not be shared with a training while this runs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config as C  # noqa: E402
from common.mot.reid import ReIDNet, crop_tensor  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--weights", default=str(C.RUNS_ROOT / "reid" / "reid_r18_kittimots.pth"))
    ap.add_argument("--min-score", type=float, default=0.1)
    a = ap.parse_args()
    device = torch.device("cuda")
    ck = torch.load(Path(a.weights).expanduser(), map_location=device, weights_only=False)
    model = ReIDNet(ck["num_ids"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    src = Path(a.dets).expanduser()
    z = np.load(src, allow_pickle=False)
    arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["meta"]))
    if meta["source"] != "kittimots":
        sys.exit("only KITTI-MOTS detections (the ReID model is trained on KITTI-MOTS)")
    root = C.DATASETS_ROOT / "KITTI_MOTS_VOC"
    seqs = [k for k in z.files if "__" not in k and k != "meta"]
    warm = torch.zeros(8, 3, ck["size"], ck["size"], device=device)
    for _ in range(20):
        model(warm)
    torch.cuda.synchronize()
    all_ms = []
    for seq in seqs:
        rows = arrays[seq]
        n_frames = len(arrays[f"{seq}__ms"])
        feat = np.zeros((len(rows), 512), np.float16)
        ms = np.zeros(n_frames, np.float32)
        for f in range(n_frames):
            sel = np.where((rows[:, 0] == f) & (rows[:, 5] >= a.min_score))[0]
            img = np.asarray(Image.open(root / "JPEGImages" / f"{seq}_{f:06d}.jpg").convert("RGB"))
            if len(sel) == 0:
                continue
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = torch.stack([crop_tensor(img, rows[i, 1:5], ck["size"]) for i in sel]).to(device)
            out = model(x)
            torch.cuda.synchronize()
            ms[f] = (time.perf_counter() - t0) * 1000
            feat[sel] = out.cpu().numpy().astype(np.float16)
        arrays[f"{seq}__feat"] = feat
        arrays[f"{seq}__reid_ms"] = ms
        all_ms.append(ms)
        print(f"  {seq}: {n_frames} frames, {int((rows[:, 5] >= a.min_score).sum())} crops, "
              f"reid {np.median(ms):.1f} ms median", flush=True)
    ms = np.concatenate(all_ms)
    meta.update({"reid_weights": str(a.weights), "reid_min_score": a.min_score,
                 "reid_ms_median": float(np.median(ms)), "reid_ms_p95": float(np.percentile(ms, 95)),
                 "reid_val": ck.get("val")})
    arrays["meta"] = np.array(json.dumps(meta))
    out = src.with_name(src.stem + "_reid.npz")
    np.savez_compressed(out, **arrays)
    print(f"-> {out}  reid {meta['reid_ms_median']:.1f} ms median, {meta['reid_ms_p95']:.1f} ms p95")


if __name__ == "__main__":
    main()
