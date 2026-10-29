"""
Semantic segmentation task (FL36-Seg / VOC2007 segmentation subset), K = C + 1 classes (0 = background).

Model contract
    training  : loss, _ = model(images, {"sem": LongTensor [B, S, S]})   (255 = ignore)
    inference : model(images) -> logits [B, K, h, w], h / w proportional to the input (upsampled here)
Evaluation (original image pixels): logits of the letterboxed area -> bilinear resize to the original size ->
argmax -> confusion matrix (void 255 skipped). Reported: mIoU over all K classes (incl. background, the same
definition as the "derived semantic mIoU" of the instance models in tasks/seg_task.py), mIoU over the C object
classes, pixel accuracy. "miou" / "miou_fg" carry {"summary": {"mAP": value}} so the common runner records them
in the eval curve like detection mAP.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import config as C
from common.datasets.voc import letterbox, letterbox_params, to_tensor
from common.datasets.voc_semseg import VOCSemSeg
from common.mask_ops import miou_from_confusion, semantic_confusion
from common.progress import Progress
from common.trainer import make_loader
from common.vis import VideoWriter, class_color


class SemSegTask:
    name = "semseg"

    def __init__(self):
        self.spec = C.TASKS["semseg"]
        self.class_names = list(self.spec["classes"])

    def train_set(self, args):
        return VOCSemSeg(C.SEG_ROOT, self.spec["train_split"], args.img_size, self.class_names, train=True,
                         hflip_prob=C.PROFILE["hflip_prob"], max_samples=args.max_train, aug=C.PROFILE["image_aug"])

    def test_set(self, args):
        return VOCSemSeg(C.SEG_ROOT, self.spec["test_split"], args.img_size, self.class_names, train=False,
                         max_samples=args.max_test)

    @staticmethod
    def to_original(logits, img_size, w, h):
        """logits [K, h', w'] over the letterboxed input -> label map [h, w] (numpy int64)."""
        lg = F.interpolate(logits[None].float(), (img_size, img_size), mode="bilinear", align_corners=False)[0]
        _, px, py, rw, rh = letterbox_params(w, h, img_size)
        lg = F.interpolate(lg[None, :, py:py + rh, px:px + rw], (h, w), mode="bilinear", align_corners=False)[0]
        return lg.argmax(0).cpu().numpy()

    @torch.no_grad()
    def evaluate(self, model, dataset, args, device):
        model.eval()
        loader = make_loader(dataset, args.batch_size, args.workers, False, args.seed)
        conf = None
        t0 = time.time()
        bar = Progress(len(loader), f"{args.run_name} | test inference")
        for idx, images, _ in loader:
            logits = model(images.to(device, dtype=torch.float32))
            for k, i in enumerate(idx):
                gt = dataset.labels(i)
                pred = self.to_original(logits[k], args.img_size, gt.shape[1], gt.shape[0])
                conf = semantic_confusion(pred, gt, len(self.class_names), conf)
            bar.update()
        bar.close()
        fps = len(dataset) / max(time.time() - t0, 1e-9)
        iou, miou = miou_from_confusion(conf)
        miou_fg = float(np.nanmean(iou[1:]))
        acc = float(np.diag(conf).sum() / max(conf.sum(), 1))
        split = getattr(args, "split_label", self.spec["test_split"])
        names = ["background"] + self.class_names
        print(f"\n{self.spec['dataset'].upper()} {split} ({len(dataset)} images) | semantic segmentation")
        for n, v in zip(names, iou):
            print(f"{n:<14}{'  -   ' if np.isnan(v) else f'{v:.4f}'}")
        print(f"mIoU (incl. background) {miou:.4f}   mIoU (objects) {miou_fg:.4f}   pixel acc {acc:.4f}")
        print(f"Inference speed incl. resize: {fps:.1f} img/s")
        per = {n: (None if np.isnan(v) else float(v)) for n, v in zip(names, iou)}
        return {"metric": "mIoU (VOC confusion matrix, void ignored)", "images": len(dataset), "fps": fps,
                "miou": {"summary": {"mAP": miou}, "per_class": per}, "miou_fg": {"summary": {"mAP": miou_fg}},
                "pixel_acc": acc}

    # ------------------------------------------------------------ demos
    @torch.no_grad()
    def predict_bgr(self, model, frame_bgr, img_size, device):
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxed, _, _, _ = letterbox(pil, img_size)
        logits = model(to_tensor(boxed).unsqueeze(0).to(device))[0]
        return self.to_original(logits, img_size, pil.size[0], pil.size[1])

    def draw(self, img, lab):
        over = img.copy()
        for k in np.unique(lab):
            if k > 0:
                over[lab == k] = class_color(int(k) - 1)
        return cv2.addWeighted(over, 0.5, img, 0.5, 0)

    def demo_image(self, model, args, device, out_dir):
        src = Path(args.input or self.spec["demo_image"])
        img = cv2.imread(str(src))
        if img is None:
            raise FileNotFoundError(src)
        lab = self.predict_bgr(model, img, args.img_size, device)
        for k, n in zip(*np.unique(lab, return_counts=True)):
            print(f"  {(['background'] + self.class_names)[k]:<12} {n} px")
        out = Path(out_dir) / f"{src.stem}_{args.tag}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), self.draw(img, lab))
        print(f"Saved        : {out}")

    def demo_video(self, model, args, device, out_dir):
        src = Path(args.input or self.spec["demo_video"])
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise FileNotFoundError(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = Path(out_dir) / f"{src.stem}_{args.tag}.mp4"
        writer = VideoWriter(out, fps, (w, h))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok or (args.num_frames and n >= args.num_frames):
                break
            writer.write(self.draw(frame, self.predict_bgr(model, frame, args.img_size, device)))
            n += 1
        cap.release()
        writer.close()
        print(f"Saved        : {out}")
