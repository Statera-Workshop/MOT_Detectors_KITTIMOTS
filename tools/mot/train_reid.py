#!/usr/bin/env python3
"""
MOT stage (2026-10-23): train the shared ReID model (common/mot/reid.py) from scratch on KITTI-MOTS train crops.

    python tools/mot/train_reid.py                       # 60 epochs, ~15-30 min on the RTX 4070 Laptop
    python tools/mot/train_reid.py --epochs 2 --check    # quick run: loader counts, a few iterations, val

PK batches (P = 16 identities x K = 4 crops = 64), Adam 3.5e-4, weight decay 5e-4, linear warm-up 10 epochs, x0.1
at epoch 40, seed 777. One epoch = (#identities // P) batches. Augmentation: horizontal flip, +-10 % shift / scale.
Validation (MOTS val sequences, never used for training): per track up to 4 crops >= 10 frames apart as queries,
gallery = every other crop of the same sequence from a different frame; rank-1 and mAP, per class and overall.
Output: <study>/runs/reid/reid_r18_kittimots.pth (+ reid_log.json). The file records the crop / id counts.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config as C  # noqa: E402
from common.mot.reid import ReIDNet, SIZE, batch_hard_triplet, crop_tensor, kitti_crops  # noqa: E402

ROOT = C.DATASETS_ROOT / "KITTI_MOTS_VOC"


class Crops(torch.utils.data.Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, box, gid = self.items[i][:3]
        img = np.asarray(Image.open(path).convert("RGB"))
        rng = np.random.default_rng() if self.train else None
        x = crop_tensor(img, box, SIZE, jitter=0.1 if self.train else None, rng=rng)
        if self.train and rng.random() < 0.5:
            x = torch.flip(x, dims=[2])
        return x, gid


def pk_batches(labels, p, k, rng):
    by_id = {}
    for i, y in enumerate(labels):
        by_id.setdefault(y, []).append(i)
    ids = [y for y, v in by_id.items() if len(v) >= 2]
    rng.shuffle(ids)
    for b in range(len(ids) // p):
        idx = []
        for y in ids[b * p:(b + 1) * p]:
            v = by_id[y]
            idx += list(rng.choice(v, k, replace=len(v) < k))
        yield idx


@torch.no_grad()
def evaluate(model, items, device, workers):
    model.eval()
    dl = torch.utils.data.DataLoader(Crops(items, False), batch_size=128, num_workers=workers)
    feats = torch.cat([model(x.to(device)).cpu() for x, _ in dl])
    gid = np.array([it[2] for it in items])
    cls = np.array([it[3] for it in items])
    seq = np.array([it[4] for it in items])
    frame = np.array([it[5] for it in items])
    rng = np.random.default_rng(777)
    res = {}
    for name, cmask in (("car", cls == 0), ("pedestrian", cls == 1), ("all", np.ones(len(cls), bool))):
        r1, aps = [], []
        for g in np.unique(gid[cmask]):
            qi = np.where(gid == g)[0]
            qi = qi[np.argsort(frame[qi])]
            picked, last = [], -100
            for i in qi:
                if frame[i] - last >= 10:
                    picked.append(i)
                    last = frame[i]
            for q in rng.permutation(picked)[:4]:
                gal = np.where((seq == seq[q]) & (frame != frame[q]) & cmask)[0]
                if not (gid[gal] == gid[q]).any():
                    continue
                s = (feats[gal] @ feats[q]).numpy()
                order = np.argsort(-s)
                hit = gid[gal][order] == gid[q]
                r1.append(float(hit[0]))
                prec = np.cumsum(hit) / (np.arange(len(hit)) + 1)
                aps.append(float((prec * hit).sum() / hit.sum()))
        res[name] = {"rank1": float(np.mean(r1)) if r1 else None, "mAP": float(np.mean(aps)) if aps else None,
                     "queries": len(r1)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default=str(C.RUNS_ROOT / "reid" / "reid_r18_kittimots.pth"))
    ap.add_argument("--check", action="store_true", help="20 iterations per epoch, small val")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda")
    train, gids = kitti_crops(ROOT, "train")
    val, vids = kitti_crops(ROOT, "val")
    labels = [it[2] for it in train]
    counts = {"train_crops": len(train), "train_ids": len(gids), "val_crops": len(val), "val_ids": len(vids),
              "train_car_crops": sum(it[3] == 0 for it in train),
              "train_pedestrian_crops": sum(it[3] == 1 for it in train)}
    print("counts:", counts, flush=True)
    model = ReIDNet(len(gids)).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=5e-4)
    ds = Crops(train, True)
    log = []
    t_start = time.time()
    for ep in range(1, a.epochs + 1):
        lr = a.lr * (ep / 10 if ep <= 10 else (1.0 if ep <= 40 else 0.1))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        batches = list(pk_batches(labels, a.p, a.k, rng))
        if a.check:
            batches = batches[:20]
        dl = torch.utils.data.DataLoader(ds, batch_sampler=batches, num_workers=a.workers)
        tot, n, t0 = 0.0, 0, time.time()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            f, logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1) + batch_hard_triplet(f, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, n = tot + float(loss), n + 1
        log.append({"epoch": ep, "loss": tot / max(n, 1), "lr": lr, "sec": time.time() - t0})
        print(f"Epoch {ep:03d} done  loss={tot / max(n, 1):.4f}  lr={lr:.2e}  time={time.time() - t0:.1f}s", flush=True)
    vres = evaluate(model, val if not a.check else val[:3000], device, a.workers)
    print("val:", json.dumps(vres), flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "num_ids": len(gids), "size": SIZE, "counts": counts,
                "val": vres, "epochs": a.epochs, "seed": a.seed, "check": a.check}, out)
    out.with_name("reid_log.json").write_text(json.dumps({"counts": counts, "log": log, "val": vres,
                                                          "minutes": (time.time() - t_start) / 60}, indent=1))
    print(f"-> {out}  ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
