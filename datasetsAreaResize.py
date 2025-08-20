import argparse
import pickle
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F   # NEW
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# --------------------- IO --------------------- #

def _unpickle(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


# --------------------- helpers --------------------- #

def _infer_tile_size_from_labels(labels: np.ndarray) -> int:
    """labels shape expected [N, H, W]"""
    assert labels.ndim == 3, f"labels ndim={labels.ndim}, expected 3"
    h, w = labels.shape[1], labels.shape[2]
    assert h == w, f"labels must be square, got {h}x{w}"
    return int(h)


def _rand_crop_xy(h: int, w: int, crop: int) -> Tuple[int, int]:
    if crop == h == w:
        return 0, 0
    if crop > h or crop > w:
        raise ValueError(f"crop_size {crop} exceeds image size {h}x{w}")
    return random.randint(0, h - crop), random.randint(0, w - crop)


def _crop_hw(arr: np.ndarray, x: int, y: int, c: int) -> np.ndarray:
    if arr.ndim == 3:
        return arr[:, x:x+c, y:y+c]
    elif arr.ndim == 2:
        return arr[x:x+c, y:y+c]
    else:
        raise ValueError(f"Unexpected array ndim={arr.ndim}")


def _has_unlabeled(a: np.ndarray) -> bool:
    return np.any(a == -1)


def _fire_prop(mask: np.ndarray) -> float:
    m = (mask > 0).astype(np.uint8)
    area = m.size
    return float(m.sum()) / float(area) if area > 0 else 0.0


def _pad_to_multiple(arr: np.ndarray, multiple: int = 32) -> np.ndarray:
    """Pad H,W up to next multiple with reflect-padding. Works for [C,H,W] or [H,W]."""
    if arr.ndim == 3:
        C, H, W = arr.shape
    elif arr.ndim == 2:
        H, W = arr.shape
    else:
        raise ValueError(f"Unexpected ndim {arr.ndim}")

    newH = ((H + multiple - 1) // multiple) * multiple
    newW = ((W + multiple - 1) // multiple) * multiple
    padH, padW = newH - H, newW - W
    if padH == 0 and padW == 0:
        return arr

    t = torch.from_numpy(arr.copy())
    if arr.ndim == 3:
        t = t.unsqueeze(0)  # [1,C,H,W]
    else:
        t = t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    padded = F.pad(t, (0, padW, 0, padH), mode="reflect")

    if arr.ndim == 3:
        padded = padded.squeeze(0).numpy()
    else:
        padded = padded.squeeze(0).squeeze(0).numpy()
    return padded


# --------------------- Dataset --------------------- #

class WildfireDataset(Dataset):
    def __init__(
        self,
        data_filename: str,
        labels_filename: str,
        features: Optional[List[int]] = None,
        crop_size: Optional[int] = None,
        random_crop: bool = False,
        fire_aware_min_prop: Optional[float] = None,
        rotate: bool = False,
        seed: int = 1,
        max_fire_aware_tries: int = 25,
    ):
        super().__init__()
        self.data = _unpickle(data_filename)        # [N, 19, H, W]
        self.labels = _unpickle(labels_filename)    # [N, H, W]

        assert self.data.ndim == 4
        assert self.labels.ndim == 3
        assert self.data.shape[0] == self.labels.shape[0]
        assert self.data.shape[2] == self.labels.shape[1] and self.data.shape[3] == self.labels.shape[2]

        self.N, self.C, self.H, self.W = self.data.shape
        self.tile_size = _infer_tile_size_from_labels(self.labels)

        self.features = sorted(features) if features is not None else None
        self.crop_size = crop_size if crop_size is not None else self.tile_size
        self.random_crop = bool(random_crop)
        self.fire_aware_min_prop = fire_aware_min_prop
        self.max_fire_aware_tries = int(max_fire_aware_tries)

        self.rotate = bool(rotate)
        self._rots = [0, 90, 180, 270]

        random.seed(seed)

        self.crop_map = self._build_crop_map()
        self.good_indices = self._compute_good_indices()

        print("=== WildfireDataset init ===")
        print(f" data: {self.data.shape} | labels: {self.labels.shape}")
        print(f" tile_size: {self.tile_size} | crop_size: {self.crop_size}")
        print(f" random_crop: {self.random_crop} | rotate: {self.rotate}")
        print(f" fire_aware_min_prop: {self.fire_aware_min_prop}")
        print(f" usable crops (no -1 after crop): {len(self.good_indices)} / {self.N}")

    def _build_crop_map(self) -> np.ndarray:
        cm = np.zeros((self.N, 2), dtype=np.int32)
        if self.random_crop and (self.crop_size < self.tile_size):
            for i in range(self.N):
                x, y = _rand_crop_xy(self.H, self.W, self.crop_size)
                cm[i] = (x, y)
        return cm

    def _compute_good_indices(self) -> np.ndarray:
        good = []
        c = self.crop_size
        for i in range(self.N):
            x, y = self.crop_map[i]
            lab = _crop_hw(self.labels[i], x, y, c)
            if not _has_unlabeled(lab):
                good.append(i)
        return np.asarray(good, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.good_indices)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.item()
        base_idx = int(self.good_indices[idx])
        x, y = self.crop_map[base_idx]
        c = self.crop_size

        if self.random_crop and (self.crop_size < self.tile_size) and (self.fire_aware_min_prop is not None):
            for _ in range(self.max_fire_aware_tries):
                x_try, y_try = _rand_crop_xy(self.H, self.W, c)
                lab_try = _crop_hw(self.labels[base_idx], x_try, y_try, c)
                if not _has_unlabeled(lab_try) and _fire_prop(lab_try) >= self.fire_aware_min_prop:
                    x, y = x_try, y_try
                    break

        feat = _crop_hw(self.data[base_idx], x, y, c)
        lab = _crop_hw(self.labels[base_idx], x, y, c)

        if self.features is not None:
            feat = feat[self.features, :, :]

        if self.rotate:
            angle = random.choice(self._rots)
            feat_t = torch.from_numpy(feat.copy())
            lab_t = torch.from_numpy(lab.copy()).unsqueeze(0)
            feat_t = TF.rotate(feat_t, angle)
            lab_t = TF.rotate(lab_t, angle)
            feat = feat_t.numpy()
            lab = lab_t.squeeze(0).numpy()

        # --- NEW: pad both feat and lab to multiple of 32 ---
        PAD_MULT = 32
        feat = _pad_to_multiple(feat, PAD_MULT)
        lab  = _pad_to_multiple(lab,  PAD_MULT)

        feat_t = torch.from_numpy(feat).float()
        lab_t  = torch.from_numpy((lab > 0).astype(np.float32)).unsqueeze(0)

        return feat_t, lab_t


def make_has_fire_weights(labels: np.ndarray) -> torch.DoubleTensor:
    N = labels.shape[0]
    has_fire = (labels.reshape(N, -1) > 0).any(axis=1)
    n_fire = int(has_fire.sum())
    n_non = N - n_fire
    w_fire = 1.0 / max(1, n_fire)
    w_non  = 1.0 / max(1, n_non)
    weights = np.where(has_fire, w_fire, w_non).astype(np.float64)
    return torch.DoubleTensor(weights)


def _demo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--random-crop", action="store_true")
    parser.add_argument("--fire-aware-min-prop", type=float, default=None)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    ds = WildfireDataset(
        data_filename=args.data,
        labels_filename=args.labels,
        crop_size=args.crop_size,
        random_crop=args.random_crop,
        fire_aware_min_prop=args.fire_aware_min_prop,
        rotate=args.rotate,
        seed=args.seed,
    )
    print(f"Dataset length: {len(ds)}")
    x, y = ds[0]
    print(f"Sample 0 -> x: {tuple(x.shape)}, y: {tuple(y.shape)}, "
          f"fire_prop: {float(y.sum().item())/y.numel():.4f}")


if __name__ == "__main__":
    _demo()
