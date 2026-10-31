# KITTI-MOTS val tracking comparison (box level, online)

## 1. HOTA, official default parameters

| detector | iou | sort | bytetrack | ocsort | hybridsort | botsort | deepsort | botsort_reid | tracktrack | fairmot | best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cascade_rcnn | 0.467 | 0.524 | 0.565 | 0.574 | 0.595 | 0.609 | 0.513 | 0.612 | **0.625** | 0.555 | tracktrack |
| libra_rcnn | 0.458 | 0.506 | 0.542 | 0.568 | 0.572 | 0.560 | 0.536 | 0.578 | **0.599** | - | tracktrack |
| faster_rcnn | 0.424 | 0.486 | 0.524 | 0.540 | 0.557 | 0.562 | 0.501 | 0.568 | **0.582** | - | tracktrack |
| dynamic_rcnn | 0.429 | 0.479 | 0.521 | 0.529 | 0.556 | 0.555 | 0.503 | 0.561 | **0.579** | 0.515 | tracktrack |
| retinanet | 0.363 | 0.398 | 0.500 | 0.435 | 0.514 | 0.527 | 0.374 | **0.535** | 0.530 | 0.488 | botsort_reid |
| atss | 0.428 | 0.474 | 0.505 | 0.472 | 0.517 | 0.511 | 0.298 | 0.519 | **0.527** | 0.515 | tracktrack |
| gfl | 0.408 | 0.428 | 0.504 | 0.454 | **0.522** | 0.514 | 0.309 | 0.521 | 0.516 | 0.511 | hybridsort |
| v6 | 0.396 | 0.432 | 0.493 | 0.445 | 0.501 | 0.505 | 0.335 | **0.509** | 0.500 | - | botsort_reid |
| tood | 0.386 | 0.410 | 0.482 | 0.427 | **0.506** | 0.495 | 0.267 | 0.503 | 0.499 | 0.485 | hybridsort |
| fcos | 0.376 | 0.399 | 0.482 | 0.413 | 0.479 | 0.490 | 0.229 | **0.493** | 0.487 | - | botsort_reid |
| ppyoloe | 0.367 | 0.408 | 0.460 | 0.414 | 0.481 | 0.480 | 0.317 | 0.484 | **0.487** | - | tracktrack |
| deformable_detr | 0.258 | 0.314 | 0.462 | 0.323 | 0.469 | 0.481 | 0.196 | **0.482** | 0.480 | - | botsort_reid |
| x | 0.361 | 0.401 | 0.443 | 0.417 | 0.465 | 0.471 | 0.277 | **0.478** | 0.468 | - | botsort_reid |
| rtmdet | 0.394 | 0.404 | 0.442 | 0.396 | **0.465** | 0.438 | 0.240 | 0.444 | 0.440 | - | hybridsort |
| v9 | 0.365 | 0.372 | 0.436 | 0.376 | 0.453 | 0.460 | 0.254 | **0.460** | 0.452 | - | botsort_reid |
| v10 | 0.357 | 0.369 | 0.441 | 0.380 | 0.452 | 0.452 | 0.229 | **0.455** | 0.450 | - | botsort_reid |
| x2 | 0.365 | 0.382 | 0.438 | 0.377 | **0.452** | 0.435 | 0.235 | 0.440 | 0.425 | - | hybridsort |
| v8 | 0.335 | 0.362 | 0.437 | 0.361 | 0.433 | 0.442 | 0.233 | **0.443** | 0.426 | - | botsort_reid |
| v11 | 0.334 | 0.354 | 0.417 | 0.357 | **0.435** | 0.429 | 0.262 | 0.429 | 0.430 | - | hybridsort |
| v12 | 0.338 | 0.360 | 0.419 | 0.347 | **0.424** | 0.416 | 0.198 | 0.416 | 0.397 | - | hybridsort |
| centernet | 0.279 | 0.302 | 0.394 | 0.267 | **0.395** | 0.344 | 0.075 | 0.344 | 0.331 | - | hybridsort |
| v26 | 0.310 | 0.322 | 0.383 | 0.301 | **0.390** | 0.337 | 0.187 | 0.337 | 0.315 | - | hybridsort |
| fairmot | 0.299 | 0.315 | 0.380 | 0.293 | **0.387** | 0.353 | 0.156 | 0.354 | 0.347 | 0.380 | hybridsort |
| v1 | 0.222 | 0.250 | 0.330 | 0.252 | 0.327 | 0.360 | 0.218 | 0.364 | **0.374** | 0.327 | tracktrack |
| v13 | 0.303 | 0.313 | 0.353 | 0.322 | 0.363 | 0.365 | 0.212 | 0.365 | **0.372** | - | tracktrack |
| ssd | 0.238 | 0.230 | 0.282 | 0.251 | 0.287 | **0.288** | 0.218 | 0.287 | 0.275 | 0.262 | botsort |
| centertrack (JDT) | - | - | - | - | - | - | - | - | - | - | 0.396 |
| trades (JDT) | - | - | - | - | - | - | - | - | - | - | 0.293 |
| motip_e61 (E2E) | - | - | - | - | - | - | - | - | - | - | 0.419 |

Causality check: all OK.

## 2. Trackers over all detectors

| tracker | detectors | mean HOTA | mean rank | wins |
|---|---|---|---|---|
| botsort_reid | 26 | 0.461 | 2.04 | 8 |
| hybridsort | 26 | 0.461 | 2.69 | 9 |
| tracktrack | 26 | 0.458 | 3.08 | 8 |
| botsort | 26 | 0.457 | 3.23 | 1 |
| bytetrack | 26 | 0.447 | 4.46 | 0 |
| fairmot | 9 | 0.449 | 5.44 | 0 |
| ocsort | 26 | 0.396 | 6.50 | 0 |
| sort | 26 | 0.384 | 7.15 | 0 |
| iou | 26 | 0.356 | 8.38 | 0 |
| deepsort | 26 | 0.284 | 9.04 | 0 |

## 3. Speed-accuracy Pareto front (RTX 4070 Laptop, batch 1, fp32, 1216 x 384)

| combination | ms / frame | HOTA |
|---|---|---|
| x2 + iou | 5.0 | 0.365 |
| x2 + sort | 5.2 | 0.382 |
| x2 + bytetrack | 5.3 | 0.438 |
| x2 + hybridsort | 5.6 | 0.452 |
| x + hybridsort | 6.1 | 0.465 |
| rtmdet + hybridsort | 6.8 | 0.465 |
| v6 + bytetrack | 6.9 | 0.493 |
| v6 + hybridsort | 7.2 | 0.501 |
| v6 + botsort | 9.5 | 0.505 |
| v6 + botsort_reid | 13.8 | 0.509 |
| gfl + hybridsort | 21.4 | 0.522 |
| retinanet + botsort | 24.0 | 0.527 |
| dynamic_rcnn + ocsort | 25.4 | 0.529 |
| faster_rcnn + ocsort | 25.4 | 0.540 |
| dynamic_rcnn + hybridsort | 25.7 | 0.556 |
| faster_rcnn + hybridsort | 25.8 | 0.557 |
| libra_rcnn + ocsort | 26.9 | 0.568 |
| libra_rcnn + hybridsort | 27.4 | 0.572 |
| faster_rcnn + tracktrack | 31.1 | 0.582 |
| libra_rcnn + tracktrack | 32.5 | 0.599 |
| cascade_rcnn + botsort | 37.8 | 0.609 |
| cascade_rcnn + botsort_reid | 40.4 | 0.612 |
| cascade_rcnn + tracktrack | 40.7 | 0.625 |

## 4. Joint / end-to-end against tracking-by-detection on the same network

| method | family | HOTA | DetA | AssA | ms | same-network TbD best | HOTA |
|---|---|---|---|---|---|---|---|
| centertrack | JDT | 0.396 | 0.341 | 0.460 | 9.6 | centernet + hybridsort | 0.395 |
| fairmot | JDT | 0.380 | 0.338 | 0.431 | 8.7 | centernet + hybridsort | 0.395 |
| trades | JDT | 0.293 | 0.238 | 0.375 | 30.4 | centernet + hybridsort | 0.395 |
| motip_e61 | E2E | 0.419 | 0.480 | 0.369 | 72.3 | deformable_detr + botsort_reid | 0.482 |

## 5. MOTIP training curve (val HOTA; the table uses the last epoch)

| epochs | 5 | 13 | 25 | 37 | 49 | 61 |
|---|---|---|---|---|---|---|
| HOTA | 0.305 | 0.366 | 0.409 | 0.408 | 0.434 | 0.419 |

## 6. Best common score threshold per detector (sensitivity; chosen on val -> not used for ranking)

| detector | iou | sort | bytetrack | ocsort | botsort | deepsort | botsort_reid |
|---|---|---|---|---|---|---|---|
| deformable_detr | 0.3 (0.361) | 0.3 (0.420) | 0.3 (0.476) | 0.3 (0.462) | 0.3 (0.510) | 0.3 (0.462) | 0.3 (0.513) |
| faster_rcnn | 0.4 (0.425) | 0.5 (0.486) | 0.7 (0.533) | 0.6 (0.540) | 0.7 (0.570) | 0.5 (0.516) | 0.7 (0.573) |
| fcos | 0.3 (0.416) | 0.3 (0.461) | 0.3 (0.503) | 0.3 (0.505) | 0.4 (0.539) | 0.3 (0.495) | 0.4 (0.542) |
| libra_rcnn | 0.5 (0.458) | 0.7 (0.508) | 0.7 (0.547) | 0.6 (0.568) | 0.7 (0.572) | 0.7 (0.539) | 0.7 (0.582) |
| ppyoloe | 0.3 (0.376) | 0.3 (0.433) | 0.4 (0.468) | 0.3 (0.480) | 0.4 (0.509) | 0.4 (0.460) | 0.4 (0.517) |
| rtmdet | 0.4 (0.413) | 0.3 (0.449) | 0.3 (0.467) | 0.3 (0.482) | 0.3 (0.510) | 0.3 (0.476) | 0.3 (0.520) |
| v10 | 0.3 (0.386) | 0.3 (0.423) | 0.3 (0.465) | 0.3 (0.461) | 0.3 (0.506) | 0.3 (0.462) | 0.3 (0.515) |
| v11 | 0.3 (0.362) | 0.3 (0.394) | 0.3 (0.445) | 0.3 (0.431) | 0.3 (0.489) | 0.3 (0.441) | 0.3 (0.498) |
| v12 | 0.3 (0.379) | 0.3 (0.412) | 0.3 (0.447) | 0.3 (0.451) | 0.3 (0.485) | 0.3 (0.448) | 0.3 (0.499) |
| v13 | 0.3 (0.325) | 0.3 (0.342) | 0.3 (0.378) | 0.3 (0.377) | 0.3 (0.413) | 0.3 (0.371) | 0.3 (0.417) |
| v26 | 0.3 (0.349) | 0.3 (0.373) | 0.3 (0.415) | 0.3 (0.403) | 0.3 (0.452) | 0.3 (0.411) | 0.3 (0.460) |
| v6 | 0.3 (0.418) | 0.3 (0.476) | 0.4 (0.509) | 0.3 (0.512) | 0.4 (0.541) | 0.3 (0.502) | 0.4 (0.543) |
| v8 | 0.3 (0.362) | 0.3 (0.403) | 0.3 (0.454) | 0.3 (0.438) | 0.4 (0.491) | 0.3 (0.437) | 0.3 (0.495) |
| v9 | 0.3 (0.395) | 0.3 (0.416) | 0.3 (0.459) | 0.3 (0.452) | 0.3 (0.490) | 0.3 (0.461) | 0.3 (0.504) |
| x | 0.3 (0.377) | 0.3 (0.421) | 0.4 (0.450) | 0.3 (0.454) | 0.3 (0.485) | 0.3 (0.438) | 0.4 (0.488) |
| x2 | 0.4 (0.376) | 0.3 (0.411) | 0.3 (0.460) | 0.4 (0.446) | 0.3 (0.487) | 0.4 (0.440) | 0.4 (0.499) |

## 7. Model size and computation

| model | kind | params (M) | GFLOPs @ 1216x384 | note |
|---|---|---|---|---|
| reid_per_crop | ReID | 11.45 | 1.2 |  |
| v26 | detector | 2.5 | 6.6 |  |
| v11 | detector | 2.59 | 7.3 |  |
| x | detector | 2.24 | 7.8 |  |
| v13 | detector | 2.46 | 8.2 |  |
| v12 | detector | 2.57 | 8.5 |  |
| v9 | detector | 2.01 | 8.8 |  |
| x2 | detector | 2.73 | 8.8 |  |
| v8 | detector | 3.01 | 9.2 |  |
| v10 | detector | 2.71 | 9.5 |  |
| v6 | detector | 4.72 | 13.8 |  |
| ppyoloe | detector | 7.61 | 18.3 |  |
| rtmdet | detector | 8.86 | 33.5 |  |
| v1 | detector | 21.28 | 43.0 |  |
| centernet | detector | 14.43 | 43.9 | lower bound (DCN / deformable attention not counted) |
| centertrack | JDT | 14.48 | 49.0 | lower bound (DCN / deformable attention not counted) |
| fairmot | JDT | 15.02 | 73.9 | lower bound (DCN / deformable attention not counted) |
| tood | detector | 32.02 | 179.2 |  |
| deformable_detr | detector | 40.1 | 180.2 | lower bound (DCN / deformable attention not counted) |
| atss | detector | 32.12 | 182.9 |  |
| fcos | detector | 32.12 | 182.9 |  |
| gfl | detector | 32.26 | 185.7 |  |
| retinanet | detector | 36.35 | 186.0 |  |
| motip | E2E | 58.53 | 187.8 | lower bound (DCN / deformable attention not counted) |
| dynamic_rcnn | detector | 41.36 | 202.9 |  |
| faster_rcnn | detector | 41.36 | 202.9 |  |
| libra_rcnn | detector | 41.62 | 207.3 |  |
| trades | JDT | 15.78 | 233.9 | lower bound (DCN / deformable attention not counted) |
| cascade_rcnn | detector | 69.15 | 258.5 |  |
| ssd | detector | 23.88 | 312.4 |  |



## v1.1 extension (2026-10-31): UCMCTrack, CAMELTrack, Seg2Track-SAM2, AB3DMOT

Four repositories added under the same protocol: `Statera-Workshop/MOT_TbD_UCMCTrack`, `MOT_Learned_CAMELTrack`,
`MOTS_SAM2_Seg2Track`, `MOT_3D_AB3DMOT` (details, caveats and all numbers in their READMEs).

### HOTA, KITTI-MOTS val (car / pedestrian mean)

| detector | best v1.0 tracker | UCMCTrack (KITTI) | CAMEL v1.1-b | Seg2Track-SAM2 |
|---|---|---|---|---|
| v1 | tracktrack 0.374 | 0.324 | 0.322 | 0.512 |
| v6 | botsort_reid 0.509 | 0.489 | 0.447 | 0.580 |
| v8 | botsort_reid 0.443 | 0.403 | 0.400 | 0.549 |
| v9 | botsort_reid 0.460 | 0.432 | 0.429 | 0.556 |
| v10 | botsort_reid 0.455 | 0.422 | 0.428 | 0.573 |
| v11 | hybridsort 0.435 | 0.404 | 0.402 | 0.579 |
| v12 | hybridsort 0.424 | 0.390 | 0.415 | 0.546 |
| v13 | tracktrack 0.372 | 0.357 | 0.356 | 0.545 |
| v26 | hybridsort 0.390 | 0.351 | 0.367 | 0.522 |
| x | botsort_reid 0.478 | 0.451 | 0.417 | 0.521 |
| x2 | hybridsort 0.452 | 0.428 | 0.424 | 0.550 |
| ppyoloe | tracktrack 0.487 | 0.466 | 0.441 | 0.558 |
| rtmdet | hybridsort 0.465 | 0.448 | 0.447 | 0.563 |
| centernet | hybridsort 0.395 | 0.339 | 0.370 | 0.558 |
| retinanet | botsort_reid 0.535 | 0.491 | 0.460 | 0.591 |
| fcos | botsort_reid 0.493 | 0.477 | 0.468 | 0.526 |
| atss | tracktrack 0.527 | 0.519 | 0.482 | 0.581 |
| gfl | hybridsort 0.522 | 0.513 | 0.494 | 0.608 |
| tood | hybridsort 0.506 | 0.480 | 0.467 | 0.591 |
| ssd | botsort 0.288 | 0.265 | 0.263 | 0.355 |
| faster_rcnn | tracktrack 0.582 | 0.563 | 0.489 | 0.504 |
| cascade_rcnn | tracktrack 0.625 | 0.600 | 0.532 | 0.606 |
| libra_rcnn | tracktrack 0.599 | 0.564 | 0.484 | 0.512 |
| dynamic_rcnn | tracktrack 0.579 | 0.548 | 0.472 | 0.556 |
| deformable_detr | botsort_reid 0.482 | 0.416 | 0.406 | 0.553 |

- **Seg2Track-SAM2** (zero-shot SAM2 tracking by segmentation): best on 21 of 25 detectors and nearly
  detector-independent - the v1.0 rule "detector > threshold > tracker" largely stops holding; about 1 s per frame,
  and KITTI-MOTS boxes are mask extents (favourable to a mask-based tracker).
- **UCMCTrack** (ground-plane tracking): with KITTI parameters chosen on train and ego-motion from oxts it comes close
  to the best v1.0 trackers at under 1 ms per frame, with metric trajectories from one camera.
- **CAMELTrack** (learned association): first attempt memorised its 10 training sequences; the corrected version gains
  +0.06-0.07 HOTA but stays below the heuristic trackers - data-limited.
- **AB3DMOT** on LiDAR detections of the OD series: Faster R-CNN heads best (BEV HOTA about 0.54); the 3D detectors saw
  about 28 % of the val frames in training (KITTI object / tracking overlap), so 3D numbers are optimistic.
