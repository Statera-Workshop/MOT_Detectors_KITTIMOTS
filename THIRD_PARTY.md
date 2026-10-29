# Third-party code

The trackers and networks are re-implemented after the official code named below (procedure, defaults); differences
are marked `[Unified]`. SORT and DeepSORT are published under GPL-3.0 by their authors; their procedures are
re-implemented here independently, no code was copied. Data: KITTI / KITTI-MOTS / KITTI-STEP, CC BY-NC-SA 3.0. This
list is informational, not legal advice.

| source | licence (as published by the source) |
| --- | --- |
| JonathonLuiten/TrackEval (installed with pip, pinned commit 12c8791) | MIT |
| rlabbe/filterpy Kalman filter equations (re-implemented) | MIT |
| bochinski/iou-tracker (procedure re-implemented) | MIT |
| abewley/sort (procedure re-implemented independently; no code copied) | GPL-3.0 (original) |
| ifzhang/ByteTrack (procedure and defaults re-implemented) | MIT |
| noahcao/OC_SORT (procedure and defaults re-implemented) | MIT |
| ymzis69/HybridSORT (procedure and defaults re-implemented) | MIT |
| NirAharon/BoT-SORT (procedure and defaults re-implemented) | MIT |
| nwojke/deep_sort (procedure re-implemented independently; no code copied) | GPL-3.0 (original) |
| kamkyu94/TrackTrack (procedure and defaults re-implemented) | MIT |
| ifzhang/FairMOT (network heads, loss, association re-implemented) | MIT |
| Legacy projects of RichardMinsooGo-ML | see the Legacy repositories |
| Megvii YOLOX (via the Legacy port) | Apache-2.0 |
| meituan/YOLOv6 tag 0.2.0 | GPL-3.0 |
| Ultralytics YOLOv5 / YOLOv8 (via the Legacy port) | AGPL-3.0 |
| ultralytics/ultralytics (official code, copied blocks) | AGPL-3.0 |
| iMoonLab/yolov13 (official code, copied blocks) | AGPL-3.0 |
| mmyolo BatchDynamicSoftLabelAssigner (via the Legacy port) | GPL-3.0 |
| PaddlePaddle/PaddleDetection 2.8 (ported to PyTorch) | Apache-2.0 |
| open-mmlab/mmdetection 3.x (RTMDet / GFL / TOOD classes) | Apache-2.0 |
| sfzhang15/ATSS (official code) | BSD-2-Clause (maskrcnn-benchmark based; check the repository) |
| xingyizhou/CenterNet (official code) | MIT |
| pytorch/vision (ResNet, SSD, DCN ops) | BSD-3-Clause |
| facebookresearch/detectron2 (RetinaNet / FCOS / FPN classes) | Apache-2.0 |
| Darknet cfg files (pjreddie/darknet, AlexeyAB/darknet) | YOLO licence (public domain) |
| fundamentalvision/Deformable-DETR (official code) | Apache-2.0 |
| facebookresearch/detr (official code) | Apache-2.0 |
