"""Evaluate the best checkpoint on the held-out test split and save prediction overlays.
    python -m src.evaluate --synthetic
    python -m src.evaluate --data_root data
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.dataset import get_datasets
from src.losses import dice_score, iou_score
from src.model import UNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--results", default="results")
    p.add_argument("--n_vis", type=int, default=6)
    args = p.parse_args()

    res = Path(args.results)
    ckpt = torch.load(res / "best_model.pth", map_location="cpu")
    a = ckpt["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=a["in_ch"], base=a["base"], dropout=a["dropout"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    _, _, test_ds, _ = get_datasets(args.data_root, args.synthetic, a["img_size"], a["seed"])
    loader = DataLoader(test_ds, batch_size=8)

    dices, ious = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)).cpu()
            dices.append(dice_score(logits, y)); ious.append(iou_score(logits, y))
    print(f"Test Dice: {sum(dices)/len(dices):.4f} | Test IoU: {sum(ious)/len(ious):.4f}")

    # Qualitative results: image | ground truth | prediction
    x, y = next(iter(DataLoader(test_ds, batch_size=args.n_vis)))
    with torch.no_grad():
        pred = (torch.sigmoid(model(x.to(device))) > 0.5).float().cpu()
    fig, ax = plt.subplots(len(x), 3, figsize=(7, 2.3 * len(x)))
    for i in range(len(x)):
        for j, (im, t) in enumerate(((x[i, 1 if x.shape[1] == 3 else 0], "MRI (FLAIR)" if x.shape[1] == 3 else "MRI"), (y[i, 0], "Ground truth"), (pred[i, 0], "Prediction"))):
            ax[i, j].imshow(im, cmap="gray"); ax[i, j].axis("off")
            if i == 0: ax[i, j].set_title(t)
    plt.tight_layout(); plt.savefig(res / "predictions.png", dpi=150)
    print("Saved", res / "predictions.png")


if __name__ == "__main__":
    main()
