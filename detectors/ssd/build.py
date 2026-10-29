"""
SSD builder (Image task only).

Paper: Liu et al., "SSD: Single Shot MultiBox Detector" (ECCV 2016, arXiv 1512.02325), SSD300 / VGG-16.
Code : torchvision 0.20.1 models/detection/ssd.py (ssd300_vgg16: _vgg_extractor = SSDFeatureExtractorVGG with
       L2-normalised conv4_3 (scale 20), dilated fc6 / 1x1 fc7, 4 extra blocks; SSDHead; compute_loss with 3:1
       hard negative mining), anchor_utils.DefaultBoxGenerator, _utils.BoxCoder (10, 10, 5, 5) and SSDMatcher (0.5).
       The authors' code is Caffe (weiliu89/caffe, branch ssd); torchvision is the approved official-equivalent
       source (2026-10-04). The classes are used directly from torchvision (installed in every environment of this
       project) instead of being copied; requirements therefore include torchvision.

    feature maps at 608 input: 76, 38, 19, 10, 8, 6 (conv4_3, fc7, extras)    default boxes 4/6/6/6/4/4 per cell
    scales 0.07 .. 1.05, ratios {2} / {2, 3}; matching IoU 0.5 + best default box per GT;
    loss smooth L1 (beta 1) on encoded boxes + softmax CE with hard negatives 3:1, both / #positives.

Raw contract: softmax over background + C classes; objectness = 1, class probabilities = softmax[1:].

[Unified] no ImageNet VGG-16 weights; common init (conv weights N(0, 0.02); L2Norm scale 20 kept); no mean / std;
608 x 608 input instead of 300 x 300, so the default-box grid follows the paper's ((i + 0.5) / |f_k|) instead of
torchvision's 300-pixel steps (8, 16, 32, 64, 100, 300); SSD's own photometric / zoom-out / min-IoU crop
augmentation is replaced by the common image augmentation; Adam 1e-4; common post-processing (instead of
score 0.01, top-400 per class, NMS 0.45, max 200).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.ssd import SSDHead, _vgg_extractor
from torchvision.ops import box_iou

from detectors.anchor_free.common import HeadStats, split_targets, to_raw


class SSD(nn.Module):
    def __init__(self, num_classes, iou_thresh=0.5, positive_fraction=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = _vgg_extractor(vgg16(weights=None), False, 5)        # all VGG layers trainable
        self.anchor_generator = DefaultBoxGenerator(
            [[2], [2, 3], [2, 3], [2, 3], [2], [2]], scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], steps=None)
        out_channels = [512, 1024, 512, 256, 256, 256]
        self.head = SSDHead(out_channels, self.anchor_generator.num_anchors_per_location(), num_classes + 1)
        self.box_coder = det_utils.BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.proposal_matcher = det_utils.SSDMatcher(iou_thresh)
        self.neg_to_pos_ratio = (1.0 - positive_fraction) / positive_fraction
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def compute_loss(self, targets, bbox_regression, cls_logits, anchors, matched_idxs):
        """torchvision SSD.compute_loss (labels: 0 = background, 1..C)"""
        num_foreground = 0
        bbox_loss, cls_targets = [], []
        for t, bbox_regression_per_image, cls_logits_per_image, anchors_per_image, matched_idxs_per_image in zip(
                targets, bbox_regression, cls_logits, anchors, matched_idxs):
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()
            matched_gt_boxes_per_image = t["boxes"][foreground_matched_idxs_per_image]
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image,
                                                             anchors_per_image[foreground_idxs_per_image, :])
            bbox_loss.append(F.smooth_l1_loss(bbox_regression_per_image[foreground_idxs_per_image, :],
                                              target_regression, reduction="sum"))
            gt_classes_target = torch.zeros((cls_logits_per_image.size(0),), dtype=torch.long,
                                            device=cls_logits_per_image.device)
            gt_classes_target[foreground_idxs_per_image] = t["labels"][foreground_matched_idxs_per_image] + 1
            cls_targets.append(gt_classes_target)
        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)
        num_classes = cls_logits.size(-1)
        cls_loss = F.cross_entropy(cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none").view(
            cls_targets.size())
        foreground_idxs = cls_targets > 0
        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float("inf")
        values, idx = negative_loss.sort(1, descending=True)
        background_idxs = idx.sort(1)[1] < num_negative
        N = max(1, num_foreground)
        losses = {"bbox_regression": bbox_loss.sum() / N,
                  "classification": (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N}
        losses["losses"] = losses["bbox_regression"] + losses["classification"]
        losses["num_pos"] = torch.tensor(num_foreground / len(targets))
        return losses, foreground_idxs.view(-1), cls_targets[foreground_idxs] - 1

    def forward(self, x, targets=None):
        features = list(self.backbone(x).values())
        head_outputs = self.head(features)
        bbox_regression, cls_logits = head_outputs["bbox_regression"], head_outputs["cls_logits"]
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        anchors = self.anchor_generator(ImageList(x, [(H, W)] * B), features)          # list of [R, 4] XYXY px
        R = bbox_regression.shape[1]
        boxes = self.box_coder.decode_single(bbox_regression.reshape(-1, 4), torch.cat(anchors, 0)).view(B, R, 4)
        raw = to_raw(boxes, F.softmax(cls_logits, -1)[..., 1:])        # objectness = 1
        if targets is None:
            return raw
        tgt = split_targets(targets, B, H)
        matched_idxs = []
        for anchors_per_image, t in zip(anchors, tgt):
            if t["boxes"].numel() == 0:
                matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64,
                                               device=anchors_per_image.device))
                continue
            matched_idxs.append(self.proposal_matcher(box_iou(t["boxes"], anchors_per_image)))
        loss_dict, pos_mask, fg_labels = self.compute_loss(tgt, bbox_regression, cls_logits, anchors, matched_idxs)
        prob = F.softmax(cls_logits.detach(), -1)[..., 1:].reshape(-1, self.num_classes)
        self.stats.update(loss_dict, prob.max(-1).values, cls_logits.detach()[..., 1:].reshape(-1, self.num_classes),
                          pos_mask, fg_labels)
        return loss_dict["losses"], raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("SSD has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    if task != "image":
        raise ValueError("SSD is implemented for the Image task first.")
    return SSD(num_classes)
