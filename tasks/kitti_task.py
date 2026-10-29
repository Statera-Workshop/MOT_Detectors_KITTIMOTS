"""
KITTI BEV tasks: Complex-YOLO ("complex") and YOLO3D ("yolo3d").

Both tasks share the dataset, BEV map, NMS, decoding, evaluation and demos.
Only the raw-output layout differs:

    complex: x y w l im re        conf cls...   (z, h fixed per class)
    yolo3d : x y z h w l im re    conf cls...

Detection contract (metric LiDAR frame, identical for both tasks):
    {"boxes": [N, 7] = x, y, z, h, w, l, yaw   (z = box bottom)
     "scores": [N], "labels": [N],
     "bev_px": [N, 6] = x, y, w, l, im, re      (BEV pixels, for drawing)}
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common import config as C
from common.datasets.kitti_bev import KittiBEVDataset
from common.geometry import bev_iou_3dboxes, iou_3d, nms_rotated
from common.kitti import config as cnf
from common.kitti.kitti_bev_utils import get_corners
from common.progress import Progress
from common.metrics import evaluate_detections, print_results
from common.trainer import make_loader
from common.vis import VideoWriter, bev_to_image, draw_bev, draw_camera_3d

LAYOUT = {
    #          BEV (x, y, w, l, im, re) cols   conf col
    "complex": ((0, 1, 2, 3, 4, 5), 6),
    "yolo3d":  ((0, 1, 4, 5, 6, 7), 8),
}


class KittiTask:
    def __init__(self, name):
        self.name = name
        self.spec = C.TASKS[name]
        self.class_names = list(self.spec["classes"])
        self.bev_cols, self.conf_col = LAYOUT[name]

    # ------------------------------------------------------------ data
    def train_set(self, args):
        return KittiBEVDataset(C.KITTI_ROOT, self.spec["train_split"], self.name, train=True,
                               hflip_prob=C.PROFILE["hflip_prob"], max_samples=args.max_train)

    def test_set(self, args):
        return KittiBEVDataset(C.KITTI_ROOT, self.spec["test_split"], self.name, train=False,
                               max_samples=args.max_test)

    # ------------------------------------------------------------ post-process
    def decode(self, rows, labels, size):
        """BEV-pixel predictions -> metric LiDAR boxes (inverse of the target encoders)."""
        bc = cnf.boundary
        dx, dy, dz = bc["maxX"] - bc["minX"], bc["maxY"] - bc["minY"], bc["maxZ"] - bc["minZ"]
        boxes = np.zeros((len(rows), 7))
        for k, (r, c) in enumerate(zip(rows, labels)):
            bx, by, bw, bl, im, re = r[list(self.bev_cols)]
            x = by / size * dx + bc["minX"]
            y = bx / size * dy + bc["minY"]
            w = bw / size * dy - cnf.BOX_WL_MARGIN
            l = bl / size * dx - cnf.BOX_WL_MARGIN
            yaw = -np.arctan2(im, re)
            if self.name == "complex":
                z, h = cnf.COMPLEX_FIXED_Z, cnf.COMPLEX_FIXED_H[int(c)]
            else:
                z = r[2] * dz + bc["minZ"]
                h = r[3] / size * dz
            boxes[k] = (x, y, z, h, max(w, 0.1), max(l, 0.1), yaw)
        return boxes

    def postprocess(self, raw, size, conf_thr):
        P = C.PROFILE
        out = []
        for pred in raw:
            pred = pred.detach().cpu()
            cls_conf, cls_id = pred[:, self.conf_col + 1:].max(1)
            score = pred[:, self.conf_col] * cls_conf
            keep = score >= conf_thr
            pred, score, cls_id = pred[keep], score[keep], cls_id[keep]
            if score.numel() > P["pre_nms_topk"]:
                top = score.topk(P["pre_nms_topk"]).indices
                pred, score, cls_id = pred[top], score[top], cls_id[top]
            pred, score, cls_id = pred.numpy().astype(np.float64), score.numpy(), cls_id.numpy()
            if len(pred):
                b = pred[:, list(self.bev_cols)]
                corners = np.stack([get_corners(x, y, w, l, np.arctan2(im, re)) for x, y, w, l, im, re in b])
                k = nms_rotated(corners, score, cls_id, P["nms_iou"])[: P["max_det"]]
            else:
                k = np.zeros(0, dtype=np.int64)
            pred, score, cls_id = pred[k], score[k], cls_id[k]
            out.append({"boxes": self.decode(pred, cls_id, size), "scores": score, "labels": cls_id,
                        "bev_px": pred[:, list(self.bev_cols)] if len(pred) else np.zeros((0, 6))})
        return out

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, model, dataset, args, device):
        model.eval()
        loader = make_loader(dataset, args.batch_size, args.workers, False, args.seed)
        dets, gts = [], []
        t0 = time.time()
        bar = Progress(len(loader), f"{args.run_name} | test inference")
        for bi, (sids, bevs, _) in enumerate(loader, 1):
            raw = model(bevs.to(device, dtype=torch.float32))
            dets += self.postprocess(raw, bevs.shape[-1], C.PROFILE["eval_conf"])
            gts += [dataset.ground_truth(s) for s in sids]
            bar.update()
        bar.close()
        fps = len(dataset) / max(time.time() - t0, 1e-9)
        thr = C.PROFILE["eval_iou"]
        bev = evaluate_detections(dets, gts, self.class_names, bev_iou_3dboxes, thr)
        box3d = evaluate_detections(dets, gts, self.class_names, iou_3d, thr)
        print_results(f"KITTI_5k {self.spec['test_split']} BEV ({len(dataset)} frames)", bev)
        print_results(f"KITTI_5k {self.spec['test_split']} 3D  ({len(dataset)} frames)", box3d)
        if self.name == "complex":
            print("Note: Complex-YOLO does not regress z/h (paper); 3D AP uses fixed class heights.")
        print(f"Inference speed incl. post-process: {fps:.1f} frames/s")
        return {"metric": "KITTI BEV/3D AP50 (single difficulty, LiDAR frame)",
                "frames": len(dataset), "fps": fps, "bev_ap50": bev, "3d_ap50": box3d}

    # ------------------------------------------------------------ demos
    @torch.no_grad()
    def predict(self, model, dataset, sid, device, conf):
        bev = torch.from_numpy(dataset.make_bev(dataset.get_lidar(sid))).float()
        raw = model(bev.unsqueeze(0).to(device))
        return bev, self.postprocess(raw, bev.shape[-1], conf)[0]

    def render(self, dataset, sid, bev, det, show_gt):
        gt = dataset.ground_truth(sid) if show_gt else None
        bev_img = draw_bev(bev_to_image(bev), det, gt)
        calib = dataset.get_calib(sid)
        cam = cv2.imread(str(dataset.image_path(sid)))
        if gt is not None:
            draw_camera_3d(cam, gt["boxes"], gt["labels"], calib, color=(255, 255, 255), thickness=1)
        draw_camera_3d(cam, det["boxes"], det["labels"], calib)
        return bev_img, cam

    def pick_sample(self, dataset, args):
        if args.sample_id:
            return args.sample_id.zfill(6)
        rng = np.random.RandomState(args.seed)
        return dataset.ids[rng.randint(len(dataset.ids))]

    def demo_image(self, model, args, device, out_dir):
        dataset = self.test_set(args)
        sid = self.pick_sample(dataset, args)
        print(f"Test frame   : {sid}  (KITTI_5k {self.spec['test_split']})")
        bev, det = self.predict(model, dataset, sid, device, args.conf)
        for b, s, l in zip(det["boxes"], det["scores"], det["labels"]):
            print(f"  {self.class_names[int(l)]:<11} {s:.3f}  xyz=({b[0]:.1f},{b[1]:.1f},{b[2]:.1f}) "
                  f"hwl=({b[3]:.1f},{b[4]:.1f},{b[5]:.1f}) yaw={b[6]:.2f}")
        bev_img, cam = self.render(dataset, sid, bev, det, args.show_gt)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for kind, img in (("bev", bev_img), ("camera3d", cam)):
            p = out_dir / f"{sid}_{args.tag}_{kind}.png"
            cv2.imwrite(str(p), img)
            print(f"Saved        : {p}")

    def demo_video(self, model, args, device, out_dir):
        dataset = self.test_set(args)
        ids = dataset.ids[: args.num_frames] if args.num_frames else dataset.ids
        print(f"Test frames  : {len(ids)} (KITTI Object frames are not temporally continuous)")
        out_dir = Path(out_dir)
        fps = args.fps
        wb = VideoWriter(out_dir / f"kitti5k_test_{args.tag}_bev.mp4", fps, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        wc = VideoWriter(out_dir / f"kitti5k_test_{args.tag}_camera3d.mp4", fps, (1242, 375))
        t0 = time.time()
        bar = Progress(len(ids), f"{args.run_name} | video")
        for n, sid in enumerate(ids, 1):
            bev, det = self.predict(model, dataset, sid, device, args.conf)
            bev_img, cam = self.render(dataset, sid, bev, det, args.show_gt)
            wb.write(bev_img)
            wc.write(cam)
            bar.update()
        bar.close()
        wb.close()
        wc.close()
        print(f"Saved        : {out_dir / f'kitti5k_test_{args.tag}_bev.mp4'}")
        print(f"Saved        : {out_dir / f'kitti5k_test_{args.tag}_camera3d.mp4'}")
