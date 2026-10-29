"""
Dynamic R-CNN builder (ResNet-50-FPN), Image task.

Paper: Zhang, Chang, Ma, Wang & Chen, "Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training"
       (ECCV 2020, arXiv 2004.06002).
Code : official hkzhang95/DynamicRCNN is maskrcnn-benchmark; the same algorithm in mmdetection 3.x is used:
       configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py, models/roi_heads/dynamic_roi_head.py
       (dynamic_rcnn: iou_topk 75, beta_topk 10, update_iter_interval 100, initial_iou 0.4, initial_beta 1.0;
       assigner pos = neg = min_pos IoU = the dynamic threshold; SmoothL1 with the dynamic beta).
       Detector, RPN and RoI head: detectors/faster_rcnn (torchvision Faster R-CNN R50-FPN).

    Dynamic Label Assignment : every 100 iterations the RoI IoU threshold becomes
                               max(0.4, mean over iterations of the 75th largest proposal IoU per image)
    Dynamic SmoothL1 Loss    : beta becomes min(1.0, median over iterations of the 10 x #img smallest |dx, dy| target)

[Unified] as detectors/faster_rcnn. The iteration counter counts training forwards (micro batches, effective batch 8
= 4 micro batches); the 100-iteration interval is kept in micro batches (mmdet counts optimizer steps).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from detectors.faster_rcnn.build import FasterRCNNDet

EPS = 1e-15


class DynamicRCNN(FasterRCNNDet):
    def __init__(self, num_classes, iou_topk=75, beta_topk=10, update_iter_interval=100, initial_iou=0.4,
                 initial_beta=1.0):
        super().__init__(num_classes)
        self.iou_topk, self.beta_topk, self.interval = iou_topk, beta_topk, update_iter_interval
        self.initial_iou, self.initial_beta = initial_iou, initial_beta
        self.cur_iou, self.cur_beta = initial_iou, initial_beta
        self.iou_history, self.beta_history, self.iters = [], [], 0
        self._set_matcher(initial_iou)

    def _set_matcher(self, iou):
        m = self.det.roi_heads.proposal_matcher
        m.high_threshold = m.low_threshold = float(iou)                 # pos = neg = min_pos IoU

    def sample(self, proposals, tgt):
        rh = self.det.roi_heads
        cur_iou = []
        for p, t in zip(proposals, tgt):                                # IoU statistics of all proposals (+ GT)
            pp = torch.cat([p, t["boxes"]]) if len(t["boxes"]) else p
            mo = box_iou(t["boxes"], pp).max(0).values if len(t["boxes"]) else pp.new_zeros(len(pp))
            k = min(self.iou_topk, len(mo))
            cur_iou.append(torch.topk(mo, k).values[-1].item())
        self.iou_history.append(float(np.mean(cur_iou)))
        proposals, _, labels, regression_targets = rh.select_training_samples(proposals, tgt)
        lab, reg = torch.cat(labels), torch.cat(regression_targets)
        pos = lab > 0
        if pos.any():
            cur_target = reg[pos, :2].abs().mean(dim=1)
            k = min(self.beta_topk * len(tgt), int(pos.sum()))
            self.beta_history.append(torch.kthvalue(cur_target, k)[0].item())
        self.iters += 1
        if self.iters % self.interval == 0:                             # update_hyperparameters
            self.cur_iou = max(self.initial_iou, float(np.mean(self.iou_history)))
            self._set_matcher(self.cur_iou)
            self.iou_history = []
            if self.beta_history and np.median(self.beta_history) >= EPS:
                self.cur_beta = min(self.initial_beta, float(np.median(self.beta_history)))
            self.beta_history = []
        return proposals, labels, regression_targets

    def box_loss(self, class_logits, box_regression, labels, regression_targets):
        labels, regression_targets = torch.cat(labels, 0), torch.cat(regression_targets, 0)
        classification_loss = F.cross_entropy(class_logits, labels)
        pos = torch.where(labels > 0)[0]
        N = class_logits.shape[0]
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
        box_loss = F.smooth_l1_loss(box_regression[pos, labels[pos]], regression_targets[pos], beta=self.cur_beta,
                                    reduction="sum") / labels.numel()
        self.stats_extra = {"dyn_iou": torch.tensor(self.cur_iou), "dyn_beta": torch.tensor(self.cur_beta)}
        return classification_loss, box_loss

    def forward(self, x, targets=None):
        out = super().forward(x, targets)
        if targets is not None and hasattr(self, "stats_extra"):
            self.stats.metrics.update({k: float(v) for k, v in self.stats_extra.items()})
        return out


def build_model(task, num_classes):
    if task != "image":
        raise ValueError("Dynamic R-CNN: Image task only.")
    return DynamicRCNN(num_classes)
