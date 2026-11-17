"""Train the U-Net.  Examples:
    python -m src.train --synthetic --epochs 5                       # pipeline test
    python -m src.train --data_root data --epochs 30                 # Kaggle LGG dataset
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import get_datasets
from src.losses import DiceBCELoss, dice_score, iou_score
from src.model import UNet


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot_loss = tot_dice = tot_iou = 0.0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot_loss += loss.item(); tot_dice += dice_score(logits, y); tot_iou += iou_score(logits, y)
    n = len(loader)
    return tot_loss / n, tot_dice / n, tot_iou / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data", help="folder containing the LGG TCGA_* .tif files (searched recursively)")
    p.add_argument("--synthetic", action="store_true", help="use generated toy data")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=32, help="base number of U-Net channels")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    out = Path(args.out); out.mkdir(exist_ok=True)

    train_ds, val_ds, test_ds, in_ch = get_datasets(args.data_root, args.synthetic, args.img_size, args.seed)
    args.in_ch = in_ch
    print(f"Split -> train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)} (in_channels={in_ch})")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, args.batch_size)

    model = UNet(in_channels=in_ch, base=args.base, dropout=args.dropout).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    hist = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tl, td, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        vl, vd, vi = run_epoch(model, val_loader, device, criterion)
        scheduler.step(vl)
        for k, v in zip(hist, (tl, vl, td, vd)):
            hist[k].append(v)
        print(f"Epoch {epoch:02d} | loss {tl:.4f}/{vl:.4f} | dice {td:.4f}/{vd:.4f} | val IoU {vi:.4f}")
        if vd > best:
            best = vd
            torch.save({"model": model.state_dict(), "args": vars(args)}, out / "best_model.pth")

    json.dump(hist, open(out / "history.json", "w"), indent=2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss")
    ax[1].plot(hist["train_dice"], label="train"); ax[1].plot(hist["val_dice"], label="val"); ax[1].set_title("Dice")
    for a in ax: a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.savefig(out / "training_curves.png", dpi=150)
    print(f"Best val Dice: {best:.4f}")


if __name__ == "__main__":
    main()
