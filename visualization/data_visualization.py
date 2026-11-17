"""Exploratory visualisation of the Kaggle LGG MRI segmentation dataset.

Usage:
    python visualization/data_visualization.py --data_root data --sample TCGA_CS_4941_19960909_14
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_pair(path):
    img = np.array(Image.open(path))
    mask = np.array(Image.open(str(path).replace(".tif", "_mask.tif")))
    return img, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--sample", default="TCGA_CS_4941_19960909_14", help="file stem of one slice")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)

    matches = list(Path(args.data_root).rglob(f"{args.sample}.tif"))
    if not matches:
        raise FileNotFoundError(f"{args.sample}.tif not found under {args.data_root}")
    path = matches[0]
    img, mask = load_pair(path)

    print("image:", img.shape, img.dtype, img.min(), img.max())
    print("mask: ", mask.shape, mask.dtype, mask.min(), mask.max())
    print("mask values:", np.unique(mask))

    # Tumour size for every slice of this patient (slices are numbered _1, _2, ...)
    patient = "_".join(path.stem.split("_")[:-1])
    slices = sorted(
        (f for f in path.parent.glob(f"{patient}_*.tif") if "_mask" not in f.name),
        key=lambda f: int(f.stem.split("_")[-1]),
    )
    print(f"\n{patient}: {len(slices)} slices")
    for f in slices:
        m = np.array(Image.open(str(f).replace(".tif", "_mask.tif")))
        print(f.name, "tumor pixels:", (m > 0).sum())

    # Figure 1: all channels + mask
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(img); axes[0].set_title("Image (3 channels)")
    for i in range(3):
        axes[i + 1].imshow(img[:, :, i], cmap="gray"); axes[i + 1].set_title(f"Channel {i}")
    axes[4].imshow(mask, cmap="gray"); axes[4].set_title("Mask")
    for ax in axes:
        ax.axis("off")
    plt.savefig(out / "viz_channels.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Figure 2: FLAIR channel, tumour overlay, intensity histograms
    flair = img[:, :, 1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(flair, cmap="gray"); axes[0].set_title("FLAIR")
    axes[1].imshow(flair, cmap="gray")
    axes[1].imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.5)
    axes[1].set_title("FLAIR + tumor mask")
    if (mask > 0).any():
        axes[2].hist(flair[mask == 0], bins=50, alpha=0.6, label="background", density=True)
        axes[2].hist(flair[mask > 0], bins=50, alpha=0.6, label="tumor", density=True)
        axes[2].legend()
    axes[2].set_title("FLAIR intensities")
    for ax in axes[:2]:
        ax.axis("off")
    plt.savefig(out / "viz_flair_overlay.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
