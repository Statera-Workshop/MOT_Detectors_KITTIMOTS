#!/usr/bin/env python3
"""
Diagnostic: can each Image model learn a small fixed subset of VOC2007 trainval?

    python tools/overfit_check.py                          # v2 v3 v4 v5, 16 images, 40 epochs
    python tools/overfit_check.py --versions v3 x --n 16 --epochs 60

Each model is trained on the first N trainval images (common protocol: Adam 1e-3, flip 0.5,
micro batch 2, accumulation 1) and then evaluated on the SAME N images (no augmentation).
A model whose layers / targets are correct reaches a high train-set mAP50; the result does not
depend on how training losses or head metrics are aggregated. Nothing is saved under runs/.
"""

import argparse
import importlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import config as C
from common.darknet import weights_init_normal
from common.datasets.voc import VOCDetection
from common.trainer import train
from common.utils import get_device, set_seed
from tasks.image_task import ImageTask

MODULE = {"v1": "yolov1", "v2": "yolov2", "v3": "yolov3", "v4": "yolov4", "v5": "yolov5", "v6": "yolov6",
          "v7": "yolov7", "v8": "yolov8", "x": "yolox", "x2": "yolox2"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", default=["v2", "v3", "v4", "v5"], choices=list(MODULE))
    ap.add_argument("--n", type=int, default=16, help="number of trainval images")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--img-size", type=int, default=C.TASKS["image"]["img_size"])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = get_device(a.device)
    rows = []
    for v in a.versions:
        set_seed(C.PROFILE["seed"])
        model = importlib.import_module(f"detectors.{MODULE[v]}.build").build_model("image", len(C.VOC_CLASSES)).to(device)
        model.apply(weights_init_normal)
        train_set = VOCDetection(C.VOC_ROOT, "trainval", a.img_size, C.VOC_CLASSES, train=True,
                                 hflip_prob=C.PROFILE["hflip_prob"], max_samples=a.n)
        eval_set = VOCDetection(C.VOC_ROOT, "trainval", a.img_size, C.VOC_CLASSES, train=False, max_samples=a.n)
        args = SimpleNamespace(batch_size=a.batch_size, workers=2, seed=C.PROFILE["seed"], accum=1, epochs=a.epochs,
                               run_name=f"OVERFIT {v}", log_interval=10 ** 9, img_size=a.img_size)
        print(f"\n==== {v}: train on {len(train_set)} trainval images, evaluate on the same images")
        t0 = time.time()
        hist = train(model, train_set, dict(C.PROFILE), args, device)
        res = ImageTask().evaluate(model, eval_set, args, device)["ap50"]["summary"]
        rows.append((v, res["mAP"], res["recall"], hist[0]["loss"], hist[-1]["loss"], time.time() - t0))
    print("\n==== train-set mAP50 after overfitting (high = the model can learn these images)")
    for v, m, r, l0, l1, t in rows:
        print(f"{v:3s}  mAP50 {m:.4f}  recall {r:.4f}  loss {l0:.3g} -> {l1:.3g}  ({t:.0f} s)")


if __name__ == "__main__":
    main()
