"""
Semantic segmentation dataset in VOC layout (FL36-Seg, VOC2007 segmentation subset).

    JPEGImages/<id>.jpg, SegmentationClass/<id>.png (0 = background, k = class k-1, 255 = void),
    ImageSets/Segmentation/<split>.txt

Output: image FloatTensor [3, S, S] (letterboxed, 0..1), targets {"sem": LongTensor [S, S]} with 255 = ignore
(void pixels and the letterbox padding). Same augmentation and random-number order as VOCDetection /
VOCInstanceSeg (HSV, scale / translate jitter, hflip), labels warped with nearest interpolation, border 255.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from common.datasets.voc import hsv_jitter, letterbox, letterbox_params, to_tensor

IGNORE = 255


def letterbox_labels(lab, size):
    h, w = lab.shape
    scale, px, py, rw, rh = letterbox_params(w, h, size)
    out = np.full((size, size), IGNORE, np.uint8)
    out[py:py + rh, px:px + rw] = cv2.resize(lab, (rw, rh), interpolation=cv2.INTER_NEAREST)
    return out


class VOCSemSeg(torch.utils.data.Dataset):
    def __init__(self, root, split, img_size, class_names, train=False, hflip_prob=0.0, max_samples=None, aug=None):
        self.root, self.img_size = Path(root), int(img_size)
        self.class_names = list(class_names)
        self.train, self.aug = train, (aug if train else None)
        self.hflip_prob = hflip_prob if train else 0.0
        split_file = self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(split_file)
        self.ids = [x.strip() for x in split_file.read_text().split() if x.strip()]
        if max_samples:
            self.ids = self.ids[:max_samples]

    def __len__(self):
        return len(self.ids)

    def image_path(self, i):
        return self.root / "JPEGImages" / f"{self.ids[i]}.jpg"

    def labels(self, i):
        lab = np.array(Image.open(self.root / "SegmentationClass" / f"{self.ids[i]}.png"))
        return lab[..., 0] if lab.ndim == 3 else lab

    def __getitem__(self, i):
        image = Image.open(self.image_path(i)).convert("RGB")
        lab = letterbox_labels(self.labels(i), self.img_size)
        image, _, _, _ = letterbox(image, self.img_size)
        if self.aug:
            arr = hsv_jitter(np.asarray(image), self.aug["hue"], self.aug["saturation"], self.aug["exposure"])
            size, j = arr.shape[0], self.aug["jitter"]
            sc = np.random.uniform(1.0 - j, 1.0 + j)               # = voc.random_scale_translate draws
            tx, ty = np.random.uniform(-j, j, 2) * size
            c = (1.0 - sc) * size / 2.0
            m = np.array([[sc, 0, c + tx], [0, sc, c + ty]], dtype=np.float32)
            arr = cv2.warpAffine(arr, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
            lab = cv2.warpAffine(lab, m, (size, size), flags=cv2.INTER_NEAREST, borderValue=IGNORE)
            image = Image.fromarray(arr)
        img = to_tensor(image)
        sem = torch.from_numpy(lab.astype(np.int64))
        if self.hflip_prob and np.random.random() < self.hflip_prob:
            img, sem = torch.flip(img, [-1]), torch.flip(sem, [-1])
        return i, img, {"sem": sem}

    @staticmethod
    def collate_fn(batch):
        idx, images, targets = zip(*batch)
        return list(idx), torch.stack(images, 0), {"sem": torch.stack([t["sem"] for t in targets], 0)}
