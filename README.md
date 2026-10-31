# MOT_Detectors_KITTIMOTS

**The 25 KITTI-MOTS detectors, the shared ReID model and the saved detections of the *Physical AI MOT* comparison.**

Every tracker of the comparison runs on these detections, so the tracker repositories (`MOT_TbD_*`) reproduce their
tables without a GPU. The detectors are the models of the *Physical AI Object Detection* series (`OD_Image_*`),
trained here on KITTI-MOTS at 1216 x 384 from scratch; this repository carries their code with the wide-input changes
(loader normalised by the canvas height, per-axis DETR box scaling, Ultralytics target scaling).

> **v1.1 (2026-10-31)**: UCMCTrack, CAMELTrack, Seg2Track-SAM2 and AB3DMOT added - summary in `RESULTS.md` (v1.1 extension).

## Detectors

| detector | params (M) | GFLOPs | val mAP50 | best tracker | HOTA |
|---|---|---|---|---|---|
| v1 | 21.28 | 43.0 | 0.5353 | tracktrack | 0.374 |
| v6 | 4.72 | 13.8 | 0.7092 | botsort_reid | 0.509 |
| v8 | 3.01 | 9.2 | 0.6553 | botsort_reid | 0.443 |
| v9 | 2.01 | 8.8 | 0.6776 | botsort_reid | 0.460 |
| v10 | 2.71 | 9.5 | 0.7038 | botsort_reid | 0.455 |
| v11 | 2.59 | 7.3 | 0.6426 | hybridsort | 0.435 |
| v12 | 2.57 | 8.5 | 0.6592 | hybridsort | 0.424 |
| v13 | 2.46 | 8.2 | 0.5835 | tracktrack | 0.372 |
| v26 | 2.5 | 6.6 | 0.6110 | hybridsort | 0.390 |
| x | 2.24 | 7.8 | 0.6502 | botsort_reid | 0.478 |
| x2 | 2.73 | 8.8 | 0.6588 | hybridsort | 0.452 |
| ppyoloe | 7.61 | 18.3 | 0.6853 | tracktrack | 0.487 |
| rtmdet | 8.86 | 33.5 | 0.6921 | hybridsort | 0.465 |
| centernet | 14.43 | 43.9 | 0.6695 | hybridsort | 0.395 |
| retinanet | 36.35 | 186.0 | 0.7252 | botsort_reid | 0.535 |
| fcos | 32.12 | 182.9 | 0.7208 | botsort_reid | 0.493 |
| atss | 32.12 | 182.9 | 0.7583 | tracktrack | 0.527 |
| gfl | 32.26 | 185.7 | 0.7602 | hybridsort | 0.522 |
| tood | 32.02 | 179.2 | 0.7407 | hybridsort | 0.506 |
| ssd | 23.88 | 312.4 | 0.6256 | botsort | 0.288 |
| faster_rcnn | 41.36 | 202.9 | 0.7543 | tracktrack | 0.582 |
| cascade_rcnn | 69.15 | 258.5 | 0.7790 | tracktrack | 0.625 |
| libra_rcnn | 41.62 | 207.3 | 0.7863 | tracktrack | 0.599 |
| dynamic_rcnn | 41.36 | 202.9 | 0.7625 | tracktrack | 0.579 |
| deformable_detr | 40.1 | 180.2 | 0.6837 | botsort_reid | 0.482 |

GFLOPs at 1216 x 384 (`tools/mot/model_stats.py`; DCN / deformable attention not counted for CenterNet and Deformable
DETR). Not in the comparison: DETR, Conditional / DAB / DN-DETR and their DC5 variants (no learning on KITTI-MOTS in 300
over-fit epochs, also at 608 square), DINO (out of memory at 1216 x 384 on 8 GB), YOLOv2-v5 / v7 / Scaled-v4 / YOLOR,
EfficientDet, RT-DETR, Sparse R-CNN (square-only code, not adapted in this round).

## Shared ReID

ResNet-18 (no pre-training) + BNNeck, cross entropy + batch-hard triplet, 128 x 128 crops of KITTI-MOTS train tracks
(car and pedestrian, 530 identities); val rank-1 0.948, mAP 0.663 (`reid/reid_log.json` in the Release).

## Release assets

`image_<v>_kittimots_e030.pth` (25), `reid_r18_kittimots.pth`, `<v>_reid.npz` (detections of KITTI-MOTS val with
detector latency and ReID features; rows `frame, x1, y1, x2, y2, score, class`), `fairmot.npz`.

## Protocol (all MOT_* repositories)

- Online / causal only: every method sees the current frame and its own state; checked by re-running prefixes of
  every sequence (the output of frame t must not change when later frames are removed).
- Input 1216 x 384 (KITTI 1242 x 375 letterboxed), detectors trained from scratch on KITTI-MOTS train (12 sequences),
  30 epochs, Adam 1e-4, effective batch 8, seed 777 - the protocol of the *Physical AI Object Detection* series.
- Metrics: HOTA / DetA / AssA (TrackEval classes, car / pedestrian mean), MOTA, IDF1, IDSW.
- Latency: RTX 4070 Laptop GPU, batch 1, fp32, medians over val; pipeline = detector (forward + post-processing) +
  ReID (appearance trackers) + tracker (CPU); image decoding / letterboxing not included.
- Trackers run with their official default parameters; frame-based values given for 30 fps are scaled to 10 fps.
  A sensitivity table with one common score threshold for every tracker is reported separately (chosen on val, so it
  is analysis, not ranking).

## Data: KITTI-MOTS (box level, val = 9 sequences, 2,981 frames)

KITTI tracking images (21 training sequences) with instance / track annotations; boxes = tight boxes of the masks,
classes car and pedestrian, ignore regions (id 10000) neither rewarded nor penalised. The annotations are taken from
the **KITTI-STEP** panoptic maps (same sequences and split, car / person instances merged from KITTI-MOTS; frames,
track counts and pedestrian masks match the KITTI-MOTS paper, car masks differ by 39 of 26,899).

```bash
bash tools/mot/get_kitti_mots.sh            # KITTI tracking images + annotations (KITTI-STEP fallback)
python tools/mot/kittimots_to_voc.py        # VOC-layout copy with tracks + count report
UNIFIED_IMAGE_DATASET=kittimots python tools/mot/count_loader.py     # counts as the loaders read them
```

**Licence of the data and of every trained weight here: KITTI / KITTI-MOTS / KITTI-STEP are CC BY-NC-SA 3.0
(non-commercial).**

## Usage

```bash
UNIFIED_IMAGE_DATASET=kittimots UNIFIED_IMAGE_H=384 python train_image_x.py --epochs 30 --lr 1e-4 --img-size 1216
UNIFIED_IMAGE_DATASET=kittimots UNIFIED_IMAGE_H=384 python tools/mot/dump_dets.py --source kittimots --version x \
    --img-size 1216 --weights runs/image_x/checkpoints/image_x_kittimots_e030.pth
python tools/mot/dump_reid.py --dets ~/mot_logs/dets/kittimots/x.npz
```

## Full comparison

See `RESULTS.md` (generated by `tools/mot/final_report.py`).

## Global Collaboration

Hills Robotics is currently participating in multiple projects in Korea and internationally.

We are actively interested in building relationships with organizations that can complement our technology with global sales networks, distribution channels, system integration capabilities, manufacturing ecosystems, AI infrastructure, robotics technologies, research capabilities, or strategic investment.

Several technology and business collaborations are already underway. Some partners, projects, system architectures, and commercial programs cannot be publicly disclosed because they are subject to confidentiality and business agreements.

We believe that the emerging Physical AI ecosystem cannot be built by a single company.
It requires a global network connecting robotics companies, AI researchers, semiconductor and computing companies, sensor manufacturers, software developers, universities, system integrators, industrial customers, investors, and global distribution partners.

We have now begun our serious exploration of the Physical AI frontier, and we welcome connections with people and organizations who want to build this ecosystem together.

## Research, Contribution, and Donation

The repositories of the Statera Working Group are intended to contribute to the broader technical discussion around Physical AI.

Researchers, engineers, developers, companies, universities, and independent contributors interested in these subjects are welcome to connect with us.

If you would like to collaborate on Physical AI or autonomous robotics, participate in research or technology development, discuss international business cooperation, propose technology or platform integration, contribute technical knowledge, or support this GitHub knowledge initiative through sponsorship or donation, please contact us.

## Contact

**Richard Minsoo Go**
Head of AI, Hills Robotics
Statera Working Group

Personal / Research Contact:
[gominsoo888@gmail.com](mailto:gominsoo888@gmail.com)

Hills Robotics / Business Contact:
[minsoo.go@hillslogis.com](mailto:minsoo.go@hillslogis.com)

Hills Robotics:
https://hillslogis.com/

For inquiries regarding next-generation outdoor autonomous robotic platforms, Physical AI Spider Grid, Intelligence Cabinet, Edge AI, or On-Premise AI architectures that are not yet publicly presented on the company website, please contact us directly.
