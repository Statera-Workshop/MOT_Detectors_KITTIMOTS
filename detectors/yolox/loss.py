"""
YOLOX criterion - Legacy YOLOX project (models/detectors/yolox/loss.py), minimal changes.

    loss = 5 * L_iou + L_obj + L_cls + L_l1        (all normalised by the number of positives)
    L_obj : BCE on every point, target = SimOTA foreground mask
    L_cls : BCE on positives, target = one-hot x IoU(pred, matched gt)
    L_iou : 1 - IoU^2 on positives
    L_l1  : L1 on the raw (tx, ty, log tw, log th) of positives

[Unified] changes against the Legacy criterion:
    - box loss 1 - GIoU  ->  1 - IoU^2  (paper "IoU loss"; official Megvii IOUloss(loss_type="iou"))
    - L1 auxiliary loss always on: in YOLOX it is enabled for the no-strong-augmentation
      stage, and the common protocol never uses strong augmentation
      (with the Legacy notebook settings it was also on from the first epoch)
    - no distributed all-reduce, no epoch argument
"""

import torch
import torch.nn.functional as F

from detectors.anchor_free.box_ops import get_ious
from detectors.yolox.simota import SimOTA


class Criterion:
    def __init__(self, num_classes, center_sampling_radius=2.5, topk_candidate=10,
                 loss_obj_weight=1.0, loss_cls_weight=1.0, loss_box_weight=5.0):
        self.num_classes = num_classes
        self.loss_obj_weight = loss_obj_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_box_weight = loss_box_weight
        self.matcher = SimOTA(num_classes, center_sampling_radius, topk_candidate)

    def loss_objectness(self, pred_obj, gt_obj):
        return F.binary_cross_entropy_with_logits(pred_obj, gt_obj, reduction="none")

    def loss_classes(self, pred_cls, gt_label):
        return F.binary_cross_entropy_with_logits(pred_cls, gt_label, reduction="none")

    def loss_bboxes(self, pred_box, gt_box):
        ious = get_ious(pred_box, gt_box, "xyxy", "iou")
        return 1.0 - ious ** 2                      # [Unified] official IOUloss("iou")

    def loss_bboxes_aux(self, pred_reg, gt_box, anchors, stride_tensors):
        gt_cxcy = (gt_box[..., :2] + gt_box[..., 2:]) * 0.5
        gt_bwbh = gt_box[..., 2:] - gt_box[..., :2]
        gt_cxcy_encode = (gt_cxcy - anchors) / stride_tensors
        gt_bwbh_encode = torch.log(gt_bwbh / stride_tensors)
        gt_box_encode = torch.cat([gt_cxcy_encode, gt_bwbh_encode], dim=-1)
        return F.l1_loss(pred_reg, gt_box_encode, reduction="none")

    def __call__(self, outputs, targets):
        """
        outputs: pred_obj / pred_cls / pred_box / pred_reg : lists of [B, M_l, *]
                 anchors, stride_tensors : lists of [M_l, 2] / [M_l, 1]; strides : [8, 16, 32]
        targets: list of {"labels": [n], "boxes": [n, 4] xyxy px}
        """
        bs = outputs["pred_cls"][0].shape[0]
        fpn_strides = outputs["strides"]
        anchors = outputs["anchors"]
        obj_preds = torch.cat(outputs["pred_obj"], dim=1)
        cls_preds = torch.cat(outputs["pred_cls"], dim=1)
        box_preds = torch.cat(outputs["pred_box"], dim=1)

        cls_targets, box_targets, obj_targets, fg_masks, fg_labels = [], [], [], [], []
        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"]
            tgt_bboxes = targets[batch_idx]["boxes"]
            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:
                num_anchors = sum(ab.shape[0] for ab in anchors)
                cls_target = obj_preds.new_zeros((0, self.num_classes))
                box_target = obj_preds.new_zeros((0, 4))
                obj_target = obj_preds.new_zeros((num_anchors, 1))
                fg_mask = obj_preds.new_zeros(num_anchors).bool()
                assigned_labels = tgt_labels.new_zeros(0)
            else:
                fg_mask, assigned_labels, assigned_ious, assigned_indexs = self.matcher(
                    fpn_strides=fpn_strides, anchors=anchors,
                    pred_obj=obj_preds[batch_idx], pred_cls=cls_preds[batch_idx],
                    pred_box=box_preds[batch_idx], tgt_labels=tgt_labels, tgt_bboxes=tgt_bboxes)
                obj_target = fg_mask.unsqueeze(-1)
                cls_target = F.one_hot(assigned_labels.long(), self.num_classes)
                cls_target = cls_target * assigned_ious.unsqueeze(-1)
                box_target = tgt_bboxes[assigned_indexs]
            cls_targets.append(cls_target)
            box_targets.append(box_target)
            obj_targets.append(obj_target)
            fg_masks.append(fg_mask)
            fg_labels.append(assigned_labels)

        cls_targets = torch.cat(cls_targets, 0)
        box_targets = torch.cat(box_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        num_fgs = fg_masks.sum().clamp(1.0)

        loss_obj = self.loss_objectness(obj_preds.view(-1, 1), obj_targets.float()).sum() / num_fgs
        cls_preds_pos = cls_preds.view(-1, self.num_classes)[fg_masks]
        loss_cls = self.loss_classes(cls_preds_pos, cls_targets).sum() / num_fgs
        box_preds_pos = box_preds.view(-1, 4)[fg_masks]
        loss_box = self.loss_bboxes(box_preds_pos, box_targets).sum() / num_fgs

        # auxiliary L1 on raw regression outputs
        reg_preds_pos = torch.cat(outputs["pred_reg"], dim=1).view(-1, 4)[fg_masks]
        anchors_pos = torch.cat(anchors, 0)[None].repeat(bs, 1, 1).view(-1, 2)[fg_masks]
        strides_pos = torch.cat(outputs["stride_tensors"], 0)[None].repeat(bs, 1, 1).view(-1, 1)[fg_masks]
        loss_box_aux = self.loss_bboxes_aux(reg_preds_pos, box_targets, anchors_pos, strides_pos).sum() / num_fgs

        losses = (self.loss_obj_weight * loss_obj + self.loss_cls_weight * loss_cls
                  + self.loss_box_weight * loss_box + loss_box_aux)
        loss_dict = dict(loss_obj=loss_obj, loss_cls=loss_cls, loss_box=loss_box,
                         loss_box_aux=loss_box_aux, losses=losses)
        return loss_dict, fg_masks, torch.cat(fg_labels, 0)
