#!/usr/bin/env python3
"""
Estimate 30-epoch FL36 training time from an overfit_check log (16 images, micro batch 2).

    python tools/estimate_time.py ~/overfit_dense.txt [--epochs 30] [--train 5011]

Calibration (2026-10-05): full epoch ~= overfit epoch time x (train / 16) x 0.7; FCOS 1.3 s -> 285 s predicted,
290 s measured. Evaluation every 5 epochs adds ~5 % (+ PreciseBN ~40 s per evaluation).
"""

import re
import sys
from collections import defaultdict

args = sys.argv[1:]
epochs = int(args[args.index("--epochs") + 1]) if "--epochs" in args else 30
n_train = int(args[args.index("--train") + 1]) if "--train" in args else 5011
times, model = defaultdict(list), None
for line in open(args[0], errors="ignore"):
    m = re.match(r"==== (\S+?):", line)
    if m:
        model = m.group(1)
    m = re.match(r"Epoch (\d+) done .*time=([0-9.]+)s", line)
    if m and model and int(m.group(1)) > 1:            # skip the warm-up epoch
        times[model].append(float(m.group(2)))
for k, v in times.items():
    ep = sum(v) / len(v) * n_train / 16 * 0.7
    print(f"{k:13s} epoch ~ {ep / 60:5.1f} min   {epochs} epochs ~ {ep * epochs * 1.05 / 3600:4.1f} h")
