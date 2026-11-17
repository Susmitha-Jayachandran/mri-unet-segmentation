"""Datasets: Kaggle LGG MRI segmentation (TCGA_*.tif + TCGA_*_mask.tif) and a synthetic toy set."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split


def find_pairs(root):
    """Find all (image, mask) pairs anywhere below `root`: X.tif <-> X_mask.tif."""
    pairs = []
    for mask_path in sorted(Path(root).rglob("*_mask.tif")):
        img_path = mask_path.with_name(mask_path.name.replace("_mask.tif", ".tif"))
        if img_path.exists():
            pairs.append((img_path, mask_path))
    if not pairs:
        raise FileNotFoundError(f"No *_mask.tif files found under '{root}'. Check --data_root.")
    return pairs


def patient_id(path):
    """TCGA_CS_4941_19960909_14.tif -> TCGA_CS_4941_19960909"""
    return "_".join(Path(path).stem.split("_")[:-1])


def split_by_patient(pairs, seed=42, val_frac=0.15, test_frac=0.15):
    """Split so that all slices of one patient land in the SAME subset (no data leakage)."""
    patients = sorted({patient_id(img) for img, _ in pairs})
    random.Random(seed).shuffle(patients)
    n = len(patients)
    n_val, n_test = max(1, round(val_frac * n)), max(1, round(test_frac * n))
    test_p = set(patients[:n_test])
    val_p = set(patients[n_test:n_test + n_val])
    train, val, test = [], [], []
    for pair in pairs:
        pid = patient_id(pair[0])
        (test if pid in test_p else val if pid in val_p else train).append(pair)
    return train, val, test


class LGGDataset(Dataset):
    """3-channel MRI slice (pre-contrast, FLAIR, post-contrast) -> binary tumour mask."""

    def __init__(self, pairs, img_size=128, augment=False):
        self.pairs, self.img_size, self.augment = pairs, img_size, augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        s = (self.img_size, self.img_size)
        img = Image.open(img_path).convert("RGB").resize(s, Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize(s, Image.NEAREST)  # nearest keeps labels binary

        img = np.asarray(img, dtype=np.float32)                  # H x W x 3
        mean, std = img.mean(axis=(0, 1), keepdims=True), img.std(axis=(0, 1), keepdims=True)
        img = ((img - mean) / (std + 1e-8)).transpose(2, 0, 1)   # per-channel z-score -> 3 x H x W
        mask = (np.asarray(mask) > 0).astype(np.float32)[None]   # 1 x H x W

        if self.augment:
            if np.random.rand() < 0.5:
                img, mask = img[:, :, ::-1], mask[:, :, ::-1]
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1, :], mask[:, ::-1, :]

        return torch.from_numpy(img.copy()), torch.from_numpy(mask.copy())


class SyntheticDataset(Dataset):
    """Noisy 1-channel images with a random bright ellipse + mask. Only for testing the pipeline."""

    def __init__(self, n=200, img_size=128, seed=0):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[:img_size, :img_size]
        self.images, self.masks = [], []
        for _ in range(n):
            cy, cx = rng.integers(30, img_size - 30, 2)
            ry, rx = rng.integers(8, 25, 2)
            mask = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1).astype(np.float32)
            img = 0.6 * mask + rng.normal(0, 0.25, (img_size, img_size)).astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-8)
            self.images.append(img.astype(np.float32))
            self.masks.append(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.images[idx]).unsqueeze(0),
                torch.from_numpy(self.masks[idx]).unsqueeze(0))


def get_datasets(data_root, synthetic, img_size, seed):
    """Returns train_ds, val_ds, test_ds, in_channels. Deterministic given `seed`."""
    if synthetic:
        full = SyntheticDataset(img_size=img_size)
        n = len(full); n_val = n_test = max(1, int(0.15 * n))
        tr, va, te = random_split(full, [n - n_val - n_test, n_val, n_test],
                                  generator=torch.Generator().manual_seed(seed))
        return tr, va, te, 1
    train, val, test = split_by_patient(find_pairs(data_root), seed)
    return (LGGDataset(train, img_size, augment=True), LGGDataset(val, img_size),
            LGGDataset(test, img_size), 3)
