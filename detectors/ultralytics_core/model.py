"""
Ultralytics-style YAML models inside the common framework (YOLOv9, YOLOv10, YOLO12, YOLO26).

parse_model() follows ultralytics/nn/tasks.py parse_model for the modules these YAMLs use
(same width / depth scaling, repeat insertion, C3k2 / A2C2f / C2fCIB legacy flags, Detect arguments,
unactivated YOLO26 SPPF rule). The detector, its head and its loss are the official ones (core.py).

UltralyticsDetector adapts them to the framework contract:
    loss, raw = model(images, targets)      raw = [B, N, cx cy w h (px), objectness=1, class probs]
    raw = model(images)
    model.yolo_layers                       logging (conf_obj / conf_noobj / cls_acc)
The dense predictions of the inference head (one-to-one for end-to-end models) go through the common
post-processing and standard NMS like every other detector ([Unified]: the same NMS for all models).
"""

import ast
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml

from detectors.anchor_free.common import HeadStats
from detectors.ultralytics_core import core as U

BASE = {U.Conv, U.SPPF, U.C2PSA, U.DWConv, U.C2f, U.C3k2, U.RepNCSPELAN4, U.ELAN1, U.ADown, U.AConv,
        U.SPPELAN, U.PSA, U.SCDown, U.C2fCIB, U.A2C2f}
REPEAT = {U.C2f, U.C3k2, U.C2PSA, U.C2fCIB, U.A2C2f}


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


EXTRA_BLOCKS = {"DSC3k2", "DSConv", "A2C2f", "HyperACE", "DownsampleConv", "FullPAD_Tunnel"}   # YOLOv13


def _module(name, extra):
    if name.startswith("nn."):
        return getattr(nn, name[3:])
    if extra is not None and name in EXTRA_BLOCKS:
        return getattr(extra, name)
    return getattr(U, name)


def parse_model(d, ch, scale, extra=None):
    """extra: a module namespace searched first (YOLOv13 blocks); its rules follow iMoonLab/yolov13 tasks.py."""
    X = extra
    x_base = {getattr(X, k) for k in ("DSC3k2", "DSConv", "A2C2f") if X is not None and hasattr(X, k)}
    x_repeat = {getattr(X, k) for k in ("DSC3k2", "A2C2f") if X is not None and hasattr(X, k)}
    legacy, max_channels = True, float("inf")
    nc, scales, end2end = d.get("nc"), d.get("scales"), d.get("end2end")
    reg_max = d.get("reg_max", 16)
    depth, width = d.get("depth_multiple", 1.0), d.get("width_multiple", 1.0)
    if scales:
        depth, width, max_channels = scales[scale]
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = _module(m, X)
        args = list(args)
        for j, a in enumerate(args):
            if isinstance(a, str):
                args[j] = {"nc": nc}.get(a, a)
                if isinstance(args[j], str):
                    try:
                        args[j] = ast.literal_eval(a)
                    except ValueError:
                        pass
        n = max(round(n * depth), 1) if n > 1 else n
        if m in x_base:                               # YOLOv13 rules (iMoonLab/yolov13 parse_model)
            c1, c2 = ch[f], make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in x_repeat:
                args.insert(2, n)
                n = 1
            if m is getattr(X, "DSC3k2", None):
                legacy = False
                if scale in "lx":
                    args[3] = True
            if m is getattr(X, "A2C2f", None):
                legacy = False
                if scale in "lx":
                    args.extend((True, 1.5))
        elif X is not None and m is getattr(X, "HyperACE", None):
            legacy = False
            c1, c2 = ch[f[1]], make_divisible(min(args[0], max_channels) * width, 8)
            he = int(args[1] * 0.5) if scale in "n" else int(args[1] * 1.5) if scale in "x" else args[1]
            args = [c1, c2, n, he, *args[2:]]
            n = 1
            if scale in "lx":
                args.append(False)
        elif X is not None and m is getattr(X, "DownsampleConv", None):
            c1 = ch[f]
            c2, args = c1 * 2, [c1]
            if scale in "lx":
                args.append(False)
                c2 = c1
        elif X is not None and m is getattr(X, "FullPAD_Tunnel", None):
            c2 = ch[f[0]]
        elif m in BASE:
            c1, c2 = ch[f], make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in REPEAT:
                args.insert(2, n)
                n = 1
            if m is U.C3k2:
                legacy = False
                if scale in {"m", "l", "x"}:
                    args[3:4] = [True]
            if m is U.A2C2f:
                legacy = False
                if scale in {"l", "x"}:
                    args.extend((True, 1.2))
            if m is U.C2fCIB:
                legacy = False
        elif m is U.Concat or (X is not None and m is getattr(X, "Concat", None)):
            c2 = sum(ch[x] for x in f)
        elif m is U.Detect:
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            m.legacy = legacy
        elif m is U.v10Detect:
            args.append([ch[x] for x in f])
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        if m is U.SPPF and len(args) <= 3:          # legacy YAML rows predate the unactivated YOLO26 SPPF
            for block in m_ if n > 1 else [m_]:
                block.cv1.act = U.Conv.default_act
        m_.i, m_.f = i, f
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class BufferDFL(nn.Module):
    """[Unified] the official DFL (a frozen 1x1 Conv2d holding 0..reg_max-1) as an identical buffer: the
    common normal(0, 0.02) initialisation re-initialises every Conv2d and would erase the projection."""

    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1
        self.register_buffer("proj", torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1))

    def forward(self, x):
        b, _, a = x.shape
        return (x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1) * self.proj).sum(1).view(b, 4, a)


class UltralyticsDetector(nn.Module):
    """Official YAML model + official head + official loss behind the framework contract."""

    def __init__(self, yaml_path, num_classes, scale, loss="v8", stal=True, extra=None):
        super().__init__()
        d = yaml.safe_load(Path(yaml_path).read_text())
        d["nc"] = num_classes
        self.yaml, self.num_classes, self.loss_kind, self.stal = d, num_classes, loss, stal
        self.model, self.save = parse_model(d, 3, scale, extra)
        head = self.model[-1]
        head.inplace = True
        if isinstance(head.dfl, U.DFL):
            head.dfl = BufferDFL(head.reg_max)
        with torch.no_grad():                         # strides as in DetectionModel.__init__ (256 px probe)
            self.train()
            out = self._forward_feats(torch.zeros(1, 3, 256, 256))
            feats = (out["one2many"] if "one2many" in out else out)["feats"]
        head.stride = torch.tensor([256 / f.shape[-2] for f in feats])
        self.stride = head.stride
        head.bias_init()
        # hyper-parameters read by the official losses (ultralytics/cfg/default.yaml); epochs set by set_epoch
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=10)
        self.criterion, self._epoch = None, 0
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    # -------------------------------------------------------------------------- graph
    def _forward_feats(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def _criterion(self):
        if self.criterion is None:                    # created lazily on the right device, as Ultralytics does
            if self.loss_kind == "e2e_prog":          # YOLO26: ProgLoss + STAL (official E2ELoss)
                self.criterion = U.E2ELoss(self)
                for _ in range(self._epoch):
                    self.criterion.update()
            elif self.loss_kind == "e2e_dual":        # YOLOv10 paper: one-to-many top-10 + one-to-one top-1
                self.criterion = U.E2EDetectLoss(self)
            else:
                self.criterion = U.v8DetectionLoss(self)
            if not self.stal:                         # [Unified] STAL (YOLO26) disabled for the earlier models
                for c in (getattr(self.criterion, "one2many", None), getattr(self.criterion, "one2one", None),
                          self.criterion):
                    if c is not None and hasattr(c, "assigner"):
                        c.assigner.stride_val = 0
        return self.criterion

    def set_epoch(self, epoch, total=None):
        """Called by common.trainer every epoch (0-based): ProgLoss decay of YOLO26 uses it."""
        if total:
            self.args.epochs = total
        if self.criterion is not None and hasattr(self.criterion, "update"):
            while self._epoch < epoch:
                self.criterion.update()
                self._epoch += 1
        self._epoch = epoch

    # -------------------------------------------------------------------------- framework contract
    def _raw(self, preds):
        head = self.model[-1]
        p = preds["one2one"] if ("one2one" in preds and head.end2end) else preds.get("one2many", preds)
        box = head._get_decode_boxes(p)                       # [B, 4, N] px
        if head.end2end or getattr(head, "xyxy", False):      # xyxy -> cxcywh
            box = torch.cat([(box[:, :2] + box[:, 2:]) / 2, box[:, 2:] - box[:, :2]], 1)
        cls = p["scores"].sigmoid()
        raw = torch.cat([box, torch.ones_like(cls[:, :1]), cls], 1)
        return raw.permute(0, 2, 1).contiguous()

    def forward(self, x, targets=None):
        head = self.model[-1]
        if targets is None:
            training = head.training
            head.train()                                       # dense predictions, no top-k post-processing
            preds = self._forward_feats(x)
            head.train(training)
            return self._raw(preds)
        preds = self._forward_feats(x)
        bboxes = targets[:, 2:6]
        if x.shape[2] != x.shape[3]:                           # [Unified] 10-23: loader normalises by the input
            bboxes = bboxes.clone()                            # height; Ultralytics scales x by W, y by H
            bboxes[:, [0, 2]] *= x.shape[2] / x.shape[3]
        batch = {"batch_idx": targets[:, 0], "cls": targets[:, 1], "bboxes": bboxes}
        crit = self._criterion()
        loss, items = crit(preds, batch)
        loss = loss.sum()                                      # as the Ultralytics trainer (loss vector x batch size)
        with torch.no_grad():                                  # logging: one-to-many assignment
            o2m = getattr(crit, "one2many", crit)
            p = preds.get("one2many", preds) if isinstance(preds, dict) else preds
            fg, gt_idx = o2m.get_assigned_targets_and_loss(p, batch)[0][:2]
            scores = p["scores"].detach().permute(0, 2, 1).sigmoid()
            labels = torch.zeros_like(gt_idx)
            for b in range(x.shape[0]):
                cls_b = targets[targets[:, 0] == b, 1].long()
                if len(cls_b):
                    labels[b] = cls_b[gt_idx[b].clamp(max=len(cls_b) - 1)]
            flat = scores.reshape(-1, self.num_classes)
            names = getattr(o2m, "loss_names", ("box", "cls", "dfl"))
            self.stats.update({f"loss_{k}": v for k, v in zip(names, items.values() if isinstance(items, dict) else items)},
                              flat.max(-1).values, torch.logit(flat.clamp(1e-6, 1 - 1e-6)),
                              fg.reshape(-1), labels.reshape(-1)[fg.reshape(-1)])
        return loss, self._raw(preds)

    def load_darknet_weights(self, path):
        raise NotImplementedError("Ultralytics-style models train from scratch here.")


def _accepts_total(fn):
    return len(inspect.signature(fn).parameters) >= 2
