"""
PP-YOLOE builder (PP-YOLOE-s: CRN-s, width 0.50, depth 0.33), Image task only.

Paper: Xu et al., "PP-YOLOE: An evolved version of YOLO" (arXiv 2203.16250, 2022).
Code : official PaddlePaddle/PaddleDetection release/2.8 (Paddle), ported class by class to PyTorch:
       configs/ppyoloe/_base_/ppyoloe_crn.yml, ppyoloe_crn_s_300e_coco.yml
       ppdet/modeling/backbones/cspresnet.py   ConvBNLayer, RepVggBlock, BasicBlock, EffectiveSELayer,
                                               CSPResStage, CSPResNet (use_large_stem, swish)
       ppdet/modeling/necks/custom_pan.py      SPP (5, 9, 13), CSPStage, CustomCSPPAN (block_num 3, spp, no DropBlock)
       ppdet/modeling/heads/ppyoloe_head.py    ESEAttn, PPYOLOEHead (reg_max 16, fixed proj, VFL alpha 0.75 gamma 2,
                                               GIoU 2.5, DFL 0.5, grid_cell_scale 5.0, offset 0.5)
       ppdet/modeling/assigners/task_aligned_assigner.py (topk 13, alpha 1, beta 6) + utils.py
       (gather_topk_anchors, check_points_inside_bboxes, compute_max_iou_anchor, generate_anchors_for_grid_cell);
       static ATSSAssigner (topk 9) via detectors/gfl (same algorithm), assigned scores = one-hot x IoU(pred, gt).
       Paddle BatchNorm2D (momentum 0.9 = torch 0.1, eps 1e-5) -> nn.BatchNorm2d; sync_bn -> BN.

[Unified] static_assigner_epoch 100 of 300 scaled to 10 of 30 (same 1/3 of the schedule); no ImageNet CSPResNet
weights; common init (nn.Conv2d / BN: pred_cls / pred_reg zero weights and ESE fc std 0.001 replaced, biases
(cls prior, reg 1.0) kept, the fixed DFL projection is a buffer); no mean / std; Mosaic-free common augmentation;
608 input; Adam 1e-4 instead of SGD 0.01 cosine + EMA; common post-processing (instead of score 0.01, top-1000,
NMS 0.7, max 300). PreciseBN (BN everywhere, micro batch 2).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors.anchor_free.common import HeadStats, split_targets, to_raw
from detectors.gfl.build import atss_assign
from detectors.rtmdet.build import bbox_overlaps

REG_MAX = 16
FPN_STRIDES = (32, 16, 8)            # the head receives P5, P4, P3


def act_fn(act):
    return nn.SiLU(inplace=True) if act == "swish" else (nn.Identity() if act is None else nn.ReLU(inplace=True))


class ConvBNLayer(nn.Module):
    def __init__(self, ch_in, ch_out, filter_size=3, stride=1, groups=1, padding=0, act=None):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, filter_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.1, eps=1e-5)
        self.act = act_fn(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(ch_in, ch_out, 1, stride=1, padding=0, act=None)
        self.act = act_fn(act)

    def forward(self, x):
        return self.act(self.conv1(x) + self.conv2(x))


class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu", shortcut=True):
        super().__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.shortcut else y


class EffectiveSELayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.Hardsigmoid()          # paddle hardsigmoid in get_act_fn: slope 1/6, offset 0.5 (= torch)

    def forward(self, x):
        return x * self.act(self.fc(x.mean((2, 3), keepdim=True)))


class CSPResStage(nn.Module):
    def __init__(self, ch_in, ch_out, n, stride, act="relu"):
        super().__init__()
        ch_mid = (ch_in + ch_out) // 2
        self.conv_down = ConvBNLayer(ch_in, ch_mid, 3, stride=2, padding=1, act=act) if stride == 2 else None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[BasicBlock(ch_mid // 2, ch_mid // 2, act=act, shortcut=True) for _ in range(n)])
        self.attn = EffectiveSELayer(ch_mid)
        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y = torch.cat([self.conv1(x), self.blocks(self.conv2(x))], 1)
        return self.conv3(self.attn(y))


class CSPResNet(nn.Module):
    def __init__(self, layers=(3, 6, 6, 3), channels=(64, 128, 256, 512, 1024), act="swish", return_idx=(1, 2, 3),
                 width_mult=1.0, depth_mult=1.0):
        super().__init__()
        channels = [max(round(c * width_mult), 1) for c in channels]
        layers = [max(round(l * depth_mult), 1) for l in layers]
        self.stem = nn.Sequential(ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act),
                                  ConvBNLayer(channels[0] // 2, channels[0] // 2, 3, stride=1, padding=1, act=act),
                                  ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act))
        self.stages = nn.Sequential(*[CSPResStage(channels[i], channels[i + 1], layers[i], 2, act=act)
                                      for i in range(len(channels) - 1)])
        self.out_channels = [channels[i + 1] for i in return_idx]
        self.return_idx = return_idx

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


class SPP(nn.Module):
    def __init__(self, ch_in, ch_out, k, pool_size, act="swish"):
        super().__init__()
        self.pool = nn.ModuleList([nn.MaxPool2d(size, 1, size // 2) for size in pool_size])
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, act=act)

    def forward(self, x):
        return self.conv(torch.cat([x] + [pool(x) for pool in self.pool], 1))


class CSPStage(nn.Module):
    def __init__(self, ch_in, ch_out, n, act="swish", spp=False):
        super().__init__()
        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        convs = []
        for i in range(n):
            convs.append(BasicBlock(ch_mid, ch_mid, act=act, shortcut=False))
            if i == (n - 1) // 2 and spp:
                convs.append(SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
        self.convs = nn.Sequential(*convs)
        self.conv3 = ConvBNLayer(ch_mid * 2, ch_out, 1, act=act)

    def forward(self, x):
        return self.conv3(torch.cat([self.conv1(x), self.convs(self.conv2(x))], 1))


class CustomCSPPAN(nn.Module):
    def __init__(self, in_channels, out_channels=(768, 384, 192), block_num=3, act="swish", spp=True,
                 width_mult=1.0, depth_mult=1.0):
        super().__init__()
        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)
        self.num_blocks = len(in_channels)
        self.out_channels = out_channels
        in_channels = in_channels[::-1]
        fpn_stages, fpn_routes = [], []
        ch_pre = 0
        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2
            fpn_stages.append(nn.Sequential(CSPStage(ch_in, ch_out, block_num, act=act, spp=(spp and i == 0))))
            if i < self.num_blocks - 1:
                fpn_routes.append(ConvBNLayer(ch_out, ch_out // 2, 1, 1, padding=0, act=act))
            ch_pre = ch_out
        self.fpn_stages, self.fpn_routes = nn.ModuleList(fpn_stages), nn.ModuleList(fpn_routes)
        pan_stages, pan_routes = [], []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(ConvBNLayer(out_channels[i + 1], out_channels[i + 1], 3, stride=2, padding=1, act=act))
            pan_stages.append(nn.Sequential(CSPStage(out_channels[i] + out_channels[i + 1], out_channels[i], block_num,
                                                     act=act, spp=False)))
        self.pan_stages, self.pan_routes = nn.ModuleList(pan_stages[::-1]), nn.ModuleList(pan_routes[::-1])

    def forward(self, blocks):
        blocks = blocks[::-1]
        fpn_feats, route = [], None
        for i, block in enumerate(blocks):
            if i > 0:
                block = torch.cat([route, block], 1)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)
            if i < self.num_blocks - 1:
                route = F.interpolate(self.fpn_routes[i](route), scale_factor=2.0)
        pan_feats = [fpn_feats[-1]]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            route = self.pan_routes[i](route)
            route = self.pan_stages[i](torch.cat([route, fpn_feats[i]], 1))
            pan_feats.append(route)
        return pan_feats[::-1]                                                   # P5, P4, P3


class ESEAttn(nn.Module):
    def __init__(self, feat_channels, act="swish"):
        super().__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)
        nn.init.normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        return self.conv(feat * torch.sigmoid(self.fc(avg_feat)))


class PPYOLOEHead(nn.Module):
    def __init__(self, in_channels, num_classes, act="swish", reg_max=REG_MAX):
        super().__init__()
        self.num_classes, self.reg_channels = num_classes, reg_max + 1
        self.stem_cls = nn.ModuleList([ESEAttn(c, act) for c in in_channels])
        self.stem_reg = nn.ModuleList([ESEAttn(c, act) for c in in_channels])
        self.pred_cls = nn.ModuleList([nn.Conv2d(c, num_classes, 3, padding=1) for c in in_channels])
        self.pred_reg = nn.ModuleList([nn.Conv2d(c, 4 * self.reg_channels, 3, padding=1) for c in in_channels])
        bias_cls = -math.log((1 - 0.01) / 0.01)
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            nn.init.constant_(cls_.weight, 0)
            nn.init.constant_(cls_.bias, bias_cls)
            nn.init.constant_(reg_.weight, 0)
            nn.init.constant_(reg_.bias, 1.0)
        # proj_conv (1x1, weight = linspace(0, reg_max), stop_gradient): a fixed buffer
        self.register_buffer("proj", torch.linspace(0, reg_max, self.reg_channels), persistent=False)

    def forward(self, feats):
        cls_list, reg_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            cls_list.append(torch.sigmoid(cls_logit).flatten(2).transpose(1, 2))
            reg_list.append(reg_distri.flatten(2).transpose(1, 2))
        return torch.cat(cls_list, 1), torch.cat(reg_list, 1)                    # [B, L, C] probs, [B, L, 68]

    def bbox_decode(self, anchor_points, pred_dist):
        B, L, _ = pred_dist.shape
        d = F.softmax(pred_dist.reshape(B, L, 4, self.reg_channels), -1) @ self.proj
        return torch.cat([anchor_points - d[..., :2], anchor_points + d[..., 2:]], -1)


def generate_anchors_for_grid_cell(sizes_hw, device, grid_cell_size=5.0, grid_cell_offset=0.5):
    anchors, points, strides = [], [], []
    for (h, w), stride in zip(sizes_hw, FPN_STRIDES):
        cell_half_size = grid_cell_size * stride * 0.5
        shift_y, shift_x = torch.meshgrid((torch.arange(h, device=device) + grid_cell_offset) * stride,
                                          (torch.arange(w, device=device) + grid_cell_offset) * stride, indexing="ij")
        shift_x, shift_y = shift_x.reshape(-1).float(), shift_y.reshape(-1).float()
        anchors.append(torch.stack([shift_x - cell_half_size, shift_y - cell_half_size,
                                    shift_x + cell_half_size, shift_y + cell_half_size], -1))
        points.append(torch.stack([shift_x, shift_y], -1))
        strides.append(torch.full((h * w, 1), float(stride), device=device))
    return anchors, torch.cat(points), torch.cat(strides)


@torch.no_grad()
def task_aligned_assign(pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, num_classes,
                        topk=13, alpha=1.0, beta=6.0, eps=1e-9):
    """ppdet TaskAlignedAssigner for one image -> labels [L], bboxes [L, 4], scores [L, C]"""
    L = pred_scores.shape[0]
    if len(gt_labels) == 0:
        return (torch.full((L,), num_classes, dtype=torch.long, device=pred_scores.device),
                pred_bboxes.new_zeros(L, 4), pred_scores.new_zeros(L, num_classes))
    ious = bbox_overlaps(gt_bboxes, pred_bboxes, eps=eps)                                # [M, L]
    alignment_metrics = pred_scores[:, gt_labels].t().pow(alpha) * ious.pow(beta)
    x, y = anchor_points[:, 0], anchor_points[:, 1]                                       # check_points_inside_bboxes
    delta = torch.stack([x[None] - gt_bboxes[:, None, 0], y[None] - gt_bboxes[:, None, 1],
                         gt_bboxes[:, None, 2] - x[None], gt_bboxes[:, None, 3] - y[None]], -1)
    is_in_gts = (delta.min(-1).values > eps).float()
    _, topk_idxs = torch.topk(alignment_metrics * is_in_gts, topk, dim=-1)               # gather_topk_anchors
    is_in_topk = F.one_hot(topk_idxs, L).sum(-2).float()
    mask_positive = is_in_topk * is_in_gts
    mask_positive_sum = mask_positive.sum(0)
    if mask_positive_sum.max() > 1:
        mask_multiple_gts = (mask_positive_sum[None] > 1).expand_as(mask_positive)
        is_max_iou = F.one_hot(ious.argmax(0), len(gt_labels)).t().float()               # compute_max_iou_anchor
        mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
        mask_positive_sum = mask_positive.sum(0)
    assigned_gt_index = mask_positive.argmax(0)
    labels = torch.where(mask_positive_sum > 0, gt_labels[assigned_gt_index],
                         torch.full_like(assigned_gt_index, num_classes))
    bboxes = gt_bboxes[assigned_gt_index]
    scores = F.one_hot(labels, num_classes + 1)[:, :num_classes].float()
    alignment_metrics = alignment_metrics * mask_positive
    max_metrics = alignment_metrics.max(-1, keepdim=True).values
    max_ious = (ious * mask_positive).max(-1, keepdim=True).values
    alignment_metrics = (alignment_metrics / (max_metrics + eps) * max_ious).max(0).values
    return labels, bboxes, scores * alignment_metrics[:, None]


class PPYOLOE(nn.Module):
    def __init__(self, num_classes, width_mult=0.50, depth_mult=0.33, static_assigner_epoch=10):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = CSPResNet(width_mult=width_mult, depth_mult=depth_mult)
        self.neck = CustomCSPPAN(self.backbone.out_channels, width_mult=width_mult, depth_mult=depth_mult)
        self.head = PPYOLOEHead(self.neck.out_channels, num_classes)
        self.static_assigner_epoch, self.epoch = static_assigner_epoch, 0
        self.precise_bn = True
        self.precise_bn_images = 1600
        self.precise_bn_batch_size = 16
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, x, targets=None):
        feats = self.neck(self.backbone(x))
        pred_scores, pred_distri = self.head(feats)
        B, C = x.shape[0], self.num_classes
        anchors_lv, anchor_points, stride_tensor = generate_anchors_for_grid_cell([f.shape[-2:] for f in feats],
                                                                                 x.device)
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.head.bbox_decode(anchor_points_s[None], pred_distri)           # stride units
        raw = to_raw(pred_bboxes * stride_tensor[None], pred_scores)
        if targets is None:
            return raw
        tgt = split_targets(targets, B, x.shape[2])
        if self.epoch < self.static_assigner_epoch:
            labels, gt_boxes = atss_assign(anchors_lv, tgt, C)
            ious = bbox_overlaps(pred_bboxes.detach() * stride_tensor[None], gt_boxes, is_aligned=True, eps=1e-9)
            scores = F.one_hot(labels, C + 1)[..., :C].float() * (ious * (labels < C)).unsqueeze(-1)
        else:
            out = [task_aligned_assign(pred_scores[b].detach(), pred_bboxes[b].detach() * stride_tensor,
                                       anchor_points, t["labels"], t["boxes"], C) for b, t in enumerate(tgt)]
            labels, gt_boxes, scores = (torch.stack(v) for v in zip(*out))
        gt_boxes = gt_boxes / stride_tensor[None]
        one_hot = F.one_hot(labels, C + 1)[..., :C].float()
        weight = 0.75 * pred_scores.pow(2.0) * (1 - one_hot) + scores * one_hot              # _varifocal_loss
        loss_cls = F.binary_cross_entropy(pred_scores, scores, weight=weight.detach(), reduction="sum")
        scores_sum = scores.sum().clamp(min=1.0)
        loss_cls = loss_cls / scores_sum
        pos = labels != C
        if pos.any():
            bbox_weight = scores.sum(-1)[pos]
            giou = bbox_overlaps(pred_bboxes[pos], gt_boxes[pos], mode="giou", is_aligned=True, eps=1e-10)
            loss_iou = ((1 - giou) * bbox_weight).sum() / scores_sum
            ltrb = torch.cat([anchor_points_s[None] - gt_boxes[..., :2], gt_boxes[..., 2:] - anchor_points_s[None]],
                             -1).clamp(0, REG_MAX - 0.01)[pos]                                  # _bbox2distance
            pd = pred_distri[pos].reshape(-1, 4, REG_MAX + 1)
            tl = ltrb.floor().long()
            wl = (tl + 1).float() - ltrb
            dfl = (F.cross_entropy(pd.reshape(-1, REG_MAX + 1), tl.reshape(-1), reduction="none").view(-1, 4) * wl +
                   F.cross_entropy(pd.reshape(-1, REG_MAX + 1), (tl + 1).reshape(-1), reduction="none").view(-1, 4)
                   * (1 - wl)).mean(-1)
            loss_dfl = (dfl * bbox_weight).sum() / scores_sum
        else:
            loss_iou = loss_dfl = pred_distri.sum() * 0
        loss_dict = {"loss_cls": 1.0 * loss_cls, "loss_iou": 2.5 * loss_iou, "loss_dfl": 0.5 * loss_dfl}
        loss_dict["losses"] = sum(loss_dict.values())
        loss_dict["num_pos"] = pos.float().sum() / B
        p = pred_scores.detach().reshape(-1, C)
        pm = pos.view(-1)
        self.stats.update(loss_dict, p.max(-1).values, torch.logit(p.clamp(1e-6, 1 - 1e-6)), pm, labels.view(-1)[pm])
        return loss_dict["losses"], raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("PP-YOLOE has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    if task != "image":
        raise ValueError("PP-YOLOE is anchor-free: Image task only.")
    return PPYOLOE(num_classes)
