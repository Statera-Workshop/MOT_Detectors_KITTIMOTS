"""
RTMDet builder (RTMDet-s by default; Image task only).

Paper: Lyu et al., "RTMDet: An Empirical Study of Designing Real-Time Object Detectors" (arXiv 2212.07784, 2022).
Code : official open-mmlab/mmdetection 3.x (configs/rtmdet/rtmdet_s_8xb32-300e_coco.py, rtmdet_l base), classes
       extracted without the mmengine / mmcv registry:
       models/layers/csp_layer.py            CSPNeXtBlock, ChannelAttention, CSPLayer
       models/backbones/cspnext.py           CSPNeXt (P5), csp_darknet.py SPPBottleneck (5, 9, 13)
       models/necks/cspnext_pafpn.py         CSPNeXtPAFPN
       models/dense_heads/rtmdet_head.py     RTMDetSepBNHead (conv weights shared across levels, separate BN,
                                             pred kernel 1, exp_on_reg False for s / tiny), RTMDetHead loss
       task_modules/assigners/dynamic_soft_label_assigner.py   DynamicSoftLabelAssigner (topk 13, iou_weight 3,
                                             soft_center_radius 3)
       losses/gfocal_loss.py (QualityFocalLoss beta 2, weight 1), iou_loss.py (GIoULoss weight 2)
       prior_generators MlvlPointGenerator (offset 0, strides 8 / 16 / 32), coders distance2bbox
       mmcv ConvModule -> Conv2d(bias=False) + BN + SiLU; DepthwiseSeparableConvModule -> dw + pw ConvModule.

[Unified] SyncBN -> BN (single GPU; SyncBN defaults momentum 0.1, eps 1e-5 kept); no ImageNet CSPNeXt weights;
common init (conv weights / BN; head prior bias kept); no mean / std; Mosaic / MixUp / large-scale jitter
replaced by the common augmentation; 608 input; Adam 1e-4 instead of AdamW 0.004 + flat-cosine + EMA; common
post-processing (instead of score 0.001, top-1000, NMS 0.65, max 300). PreciseBN (BN everywhere, micro batch 2).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors.anchor_free.common import HeadStats, split_targets, to_raw

EPS, INF = 1.0e-7, 100000000.0
STRIDES = (8, 16, 32)
VARIANTS = {   # deepen, widen, neck in, neck out / head channels, neck csp blocks
    "rtmdet_tiny": (0.167, 0.375, [96, 192, 384], 96, 1),
    "rtmdet_s": (0.33, 0.5, [128, 256, 512], 128, 1),
}
VARIANT = "rtmdet_s"


def norm(c):
    return nn.BatchNorm2d(c, momentum=0.1, eps=1e-5)          # SyncBN defaults


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = norm(out_channels)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise_conv = ConvModule(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        with torch.amp.autocast(x.device.type, enabled=False):
            out = self.global_avgpool(x)
        out = self.act(self.fc(out))
        return x * out


class CSPNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=0.5, add_identity=True, kernel_size=5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvModule(in_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv2 = DepthwiseSeparableConvModule(hidden_channels, out_channels, kernel_size, 1, kernel_size // 2)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=0.5, num_blocks=1, add_identity=True,
                 channel_attention=False):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(in_channels, mid_channels, 1)
        self.short_conv = ConvModule(in_channels, mid_channels, 1)
        self.final_conv = ConvModule(2 * mid_channels, out_channels, 1)
        self.blocks = nn.Sequential(*[CSPNeXtBlock(mid_channels, mid_channels, 1.0, add_identity)
                                      for _ in range(num_blocks)])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x):
        x_final = torch.cat((self.blocks(self.main_conv(x)), self.short_conv(x)), dim=1)
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels, mid_channels, 1)
        self.poolings = nn.ModuleList([nn.MaxPool2d(k, 1, k // 2) for k in kernel_sizes])
        self.conv2 = ConvModule(mid_channels * (len(kernel_sizes) + 1), out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        with torch.amp.autocast(x.device.type, enabled=False):
            x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        return self.conv2(x)


class CSPNeXt(nn.Module):
    arch_setting = [[64, 128, 3, True, False], [128, 256, 6, True, False], [256, 512, 6, True, False],
                    [512, 1024, 3, False, True]]                                  # P5

    def __init__(self, deepen_factor, widen_factor, out_indices=(2, 3, 4), expand_ratio=0.5,
                 channel_attention=True):
        super().__init__()
        self.out_indices = out_indices
        c0 = int(self.arch_setting[0][0] * widen_factor // 2)
        self.stem = nn.Sequential(ConvModule(3, c0, 3, padding=1, stride=2), ConvModule(c0, c0, 3, padding=1),
                                  ConvModule(c0, int(self.arch_setting[0][0] * widen_factor), 3, padding=1))
        self.layers = ["stem"]
        for i, (in_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(self.arch_setting):
            in_channels, out_channels = int(in_channels * widen_factor), int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = [ConvModule(in_channels, out_channels, 3, stride=2, padding=1)]
            if use_spp:
                stage.append(SPPBottleneck(out_channels, out_channels))
            stage.append(CSPLayer(out_channels, out_channels, expand_ratio, num_blocks, add_identity,
                                  channel_attention))
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{i + 1}")

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            x = getattr(self, layer_name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class CSPNeXtPAFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_csp_blocks=3, expand_ratio=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce_layers, self.top_down_blocks = nn.ModuleList(), nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(ConvModule(in_channels[idx], in_channels[idx - 1], 1))
            self.top_down_blocks.append(CSPLayer(in_channels[idx - 1] * 2, in_channels[idx - 1], expand_ratio,
                                                 num_csp_blocks, add_identity=False))
        self.downsamples, self.bottom_up_blocks = nn.ModuleList(), nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(ConvModule(in_channels[idx], in_channels[idx], 3, stride=2, padding=1))
            self.bottom_up_blocks.append(CSPLayer(in_channels[idx] * 2, in_channels[idx + 1], expand_ratio,
                                                  num_csp_blocks, add_identity=False))
        self.out_convs = nn.ModuleList([ConvModule(c, out_channels, 3, padding=1) for c in in_channels])

    def forward(self, inputs):
        n = len(self.in_channels)
        inner_outs = [inputs[-1]]
        for idx in range(n - 1, 0, -1):
            feat_heigh = self.reduce_layers[n - 1 - idx](inner_outs[0])
            inner_outs[0] = feat_heigh
            inner_out = self.top_down_blocks[n - 1 - idx](torch.cat([self.upsample(feat_heigh), inputs[idx - 1]], 1))
            inner_outs.insert(0, inner_out)
        outs = [inner_outs[0]]
        for idx in range(n - 1):
            outs.append(self.bottom_up_blocks[idx](torch.cat([self.downsamples[idx](outs[-1]), inner_outs[idx + 1]], 1)))
        return tuple(conv(o) for conv, o in zip(self.out_convs, outs))


class RTMDetSepBNHead(nn.Module):
    def __init__(self, num_classes, in_channels, feat_channels, stacked_convs=2, pred_kernel_size=1,
                 exp_on_reg=False, share_conv=True):
        super().__init__()
        self.exp_on_reg = exp_on_reg
        self.cls_convs, self.reg_convs = nn.ModuleList(), nn.ModuleList()
        self.rtm_cls, self.rtm_reg = nn.ModuleList(), nn.ModuleList()
        for _ in STRIDES:
            self.cls_convs.append(nn.ModuleList([ConvModule(in_channels if i == 0 else feat_channels, feat_channels,
                                                            3, 1, 1) for i in range(stacked_convs)]))
            self.reg_convs.append(nn.ModuleList([ConvModule(in_channels if i == 0 else feat_channels, feat_channels,
                                                            3, 1, 1) for i in range(stacked_convs)]))
            self.rtm_cls.append(nn.Conv2d(feat_channels, num_classes, pred_kernel_size, padding=pred_kernel_size // 2))
            self.rtm_reg.append(nn.Conv2d(feat_channels, 4, pred_kernel_size, padding=pred_kernel_size // 2))
        if share_conv:
            for n in range(len(STRIDES)):
                for i in range(stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv
        for m in self.modules():                                                   # init_weights
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        bias_cls = float(-math.log((1 - 0.01) / 0.01))                             # bias_init_with_prob(0.01)
        for rtm_cls in self.rtm_cls:
            nn.init.constant_(rtm_cls.bias, bias_cls)

    def forward(self, feats):
        cls_scores, bbox_preds = [], []
        for idx, (x, stride) in enumerate(zip(feats, STRIDES)):
            cls_feat, reg_feat = x, x
            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)
            reg = self.rtm_reg[idx](reg_feat)
            cls_scores.append(self.rtm_cls[idx](cls_feat))
            bbox_preds.append((reg.exp() if self.exp_on_reg else reg) * stride)
        return cls_scores, bbox_preds


def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
    """mmdet structures/bbox/bbox_overlaps.py (2-D case)"""
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    if is_aligned:
        lt, rb = torch.max(bboxes1[..., :2], bboxes2[..., :2]), torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        union = area1 + area2 - overlap
        enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
        enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        union = area1[..., None] + area2[..., None, :] - overlap
        enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
    union = torch.max(union, union.new_tensor([eps]))
    ious = overlap / union
    if mode == "iou":
        return ious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = torch.max(enclose_wh[..., 0] * enclose_wh[..., 1], union.new_tensor([eps]))
    return ious - (enclose_area - union) / enclose_area


class DynamicSoftLabelAssigner:
    def __init__(self, soft_center_radius=3.0, topk=13, iou_weight=3.0):
        self.soft_center_radius, self.topk, self.iou_weight = soft_center_radius, topk, iou_weight

    @torch.no_grad()
    def assign(self, pred_scores, decoded_bboxes, priors, gt_bboxes, gt_labels):
        """returns assigned_gt_inds (0 = negative, k = gt k-1) and max_overlaps"""
        num_gt, num_bboxes = gt_bboxes.size(0), decoded_bboxes.size(0)
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            return assigned_gt_inds, decoded_bboxes.new_zeros((num_bboxes,))
        prior_center = priors[:, :2]
        deltas = torch.cat([prior_center[:, None] - gt_bboxes[:, :2], gt_bboxes[:, 2:] - prior_center[:, None]], -1)
        is_in_gts = deltas.min(dim=-1).values > 0
        valid_mask = is_in_gts.sum(dim=1) > 0
        valid_decoded_bbox, valid_pred_scores = decoded_bboxes[valid_mask], pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)
        if num_valid == 0:
            return assigned_gt_inds, decoded_bboxes.new_zeros((num_bboxes,))
        gt_center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0
        valid_prior = priors[valid_mask]
        strides = valid_prior[:, 2]
        distance = (valid_prior[:, None, :2] - gt_center[None, :, :]).pow(2).sum(-1).sqrt() / strides[:, None]
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)
        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight
        gt_onehot_label = F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1]).float().unsqueeze(0).repeat(
            num_valid, 1, 1)
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores.sigmoid()
        soft_cls_cost = F.binary_cross_entropy_with_logits(valid_pred_scores, soft_label, reduction="none") * \
            scale_factor.abs().pow(2.0)
        cost_matrix = soft_cls_cost.sum(dim=-1) + iou_cost + soft_center_prior
        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(cost_matrix, pairwise_ious, num_gt, valid_mask)
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,), -INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return assigned_gt_inds, max_overlaps

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=int(dynamic_ks[gt_idx]), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1
        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds


def quality_focal_loss(pred, label, score, beta=2.0):
    """mmdet gfocal_loss.quality_focal_loss (per-sample, summed over classes)"""
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(pred, zerolabel, reduction="none") * scale_factor.pow(beta)
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(pred[pos, pos_label], score[pos],
                                                              reduction="none") * scale_factor.abs().pow(beta)
    return loss.sum(dim=1)


class RTMDet(nn.Module):
    def __init__(self, num_classes, variant=VARIANT):
        super().__init__()
        deepen, widen, neck_in, ch, csp_blocks = VARIANTS[variant]
        self.num_classes = num_classes
        self.backbone = CSPNeXt(deepen, widen)
        self.neck = CSPNeXtPAFPN(neck_in, ch, csp_blocks)
        self.head = RTMDetSepBNHead(num_classes, ch, ch)
        self.assigner = DynamicSoftLabelAssigner(topk=13)
        self.precise_bn = True                 # [Unified] common/precise_bn.py
        self.precise_bn_images = 1600
        self.precise_bn_batch_size = 16
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    @staticmethod
    def grid_priors(sizes, device):
        """MlvlPointGenerator(offset=0).grid_priors(with_stride=True) -> [R, 4] (x, y, stride, stride)"""
        out = []
        for (h, w), s in zip(sizes, STRIDES):
            sy, sx = torch.meshgrid(torch.arange(h, device=device) * s, torch.arange(w, device=device) * s,
                                    indexing="ij")
            st = torch.full_like(sx, s)
            out.append(torch.stack([sx.reshape(-1), sy.reshape(-1), st.reshape(-1), st.reshape(-1)], 1).float())
        return torch.cat(out, 0)

    def forward(self, x, targets=None):
        cls_scores, bbox_preds = self.head(self.neck(self.backbone(x)))
        B = x.shape[0]
        priors = self.grid_priors([c.shape[-2:] for c in cls_scores], x.device)
        cls = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes) for c in cls_scores], 1)
        dist = torch.cat([b.permute(0, 2, 3, 1).reshape(B, -1, 4) for b in bbox_preds], 1)
        p = priors[None, :, :2]
        boxes = torch.cat([p - dist[..., :2], p + dist[..., 2:]], -1)            # distance2bbox
        raw = to_raw(boxes, cls.sigmoid())
        if targets is None:
            return raw
        tgt = split_targets(targets, B, x.shape[2])
        R = priors.shape[0]
        labels = torch.full((B, R), self.num_classes, dtype=torch.long, device=x.device)
        assign_metrics = torch.zeros(B, R, device=x.device)
        bbox_targets = torch.zeros(B, R, 4, device=x.device)
        for b, t in enumerate(tgt):
            gt_inds, max_overlaps = self.assigner.assign(cls[b].detach(), boxes[b].detach(), priors, t["boxes"],
                                                         t["labels"])
            pos = gt_inds > 0
            if pos.any():
                labels[b, pos] = t["labels"][gt_inds[pos] - 1]
                bbox_targets[b, pos] = t["boxes"][gt_inds[pos] - 1]
                assign_metrics[b, pos] = max_overlaps[pos]
        labels_f, metrics_f = labels.view(-1), assign_metrics.view(-1)
        loss_cls = quality_focal_loss(cls.reshape(-1, self.num_classes), labels_f, metrics_f).sum()
        pos_inds = (labels_f < self.num_classes).nonzero().squeeze(1)
        if len(pos_inds) > 0:
            giou = bbox_overlaps(boxes.reshape(-1, 4)[pos_inds], bbox_targets.view(-1, 4)[pos_inds], mode="giou",
                                 is_aligned=True, eps=1e-6)
            loss_bbox = 2.0 * ((1 - giou) * metrics_f[pos_inds]).sum()
            bbox_avg = metrics_f[pos_inds].sum().clamp(min=1).item()
        else:
            loss_bbox, bbox_avg = dist.sum() * 0, 1.0
        cls_avg = metrics_f.sum().clamp(min=1).item()
        loss_dict = {"loss_cls": loss_cls / cls_avg, "loss_bbox": loss_bbox / bbox_avg}
        loss_dict["losses"] = loss_dict["loss_cls"] + loss_dict["loss_bbox"]
        loss_dict["num_pos"] = torch.tensor(len(pos_inds) / B)
        cls_flat = cls.detach().reshape(-1, self.num_classes)
        pos_mask = labels_f < self.num_classes
        self.stats.update(loss_dict, cls_flat.sigmoid().max(-1).values, cls_flat, pos_mask, labels_f[pos_mask])
        return loss_dict["losses"], raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("RTMDet has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    if task != "image":
        raise ValueError("RTMDet is anchor-free: Image task only.")
    return RTMDet(num_classes)
