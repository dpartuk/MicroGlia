import os
import time
import json
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import DPGNet, DPGNetLoss
from .dataset import get_dataloaders


def compute_metrics(preds, targets, threshold=0.5, eps=1e-6):
    """
    Compute binary Dice coefficient and Intersection over Union (IoU).

    Args:
        preds: Raw logits tensor of shape [B, 1, H, W]
        targets: Binary ground truth tensor of shape [B, 1, H, W] in {0, 1}
        threshold: Sigmoid probability threshold
        eps: Small epsilon to avoid division by zero

    Returns:
        (dice, iou) as float scalars
    """
    prob = torch.sigmoid(preds)
    pred_bin = (prob > threshold).float()

    intersection = (pred_bin * targets).sum(dim=(2, 3))
    total_pred = pred_bin.sum(dim=(2, 3))
    total_target = targets.sum(dim=(2, 3))

    dice = (2.0 * intersection + eps) / (total_pred + total_target + eps)
    union = total_pred + total_target - intersection
    iou = (intersection + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    total_b = len(dataloader)
    for b_idx, (images, masks) in enumerate(dataloader, start=1):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss, _ = criterion(preds, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # Evaluate metrics on the finest resolution prediction P5
        d, i = compute_metrics(preds[4], masks)
        running_dice += d
        running_iou += i

        if b_idx % 25 == 0 or b_idx == total_b:
            print(f"   Batch [{b_idx:03d}/{total_b:03d}] - Loss: {loss.item():.4f} - Dice: {d:.4f}", flush=True)

    num_batches = len(dataloader)
    return (
        running_loss / num_batches,
        running_dice / num_batches,
        running_iou / num_batches,
    )


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation dataset."""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss, _ = criterion(preds, masks)

        running_loss += loss.item()
        d, i = compute_metrics(preds[4], masks)
        running_dice += d
        running_iou += i

    num_batches = len(dataloader)
    return (
        running_loss / num_batches,
        running_dice / num_batches,
        running_iou / num_batches,
    )


def train_dpgnet(
    epochs=20,
    batch_size=4,
    lr=1e-4,
    weight_decay=1e-4,
    save_dir="dpgnet/output/checkpoints",
    data_root="contour/output",
    device_name=None,
    val_interval=1,
    patience=3,
    min_delta=0.001,
    resume=False,
):
    """
    Train DPGNet on microglia dataset with early stopping.

    Args:
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Initial learning rate
        weight_decay: Weight decay for AdamW
        save_dir: Destination folder for checkpoints and history
        data_root: Root folder containing slide ground truth masks
        device_name: Device string ('mps', 'cuda', 'cpu', or None for auto)
        val_interval: Epoch interval for validation
        patience: Epochs without improvement before early stopping
        min_delta: Minimum metric change to qualify as improvement
        resume: Resume from latest checkpoint if available
    """
    os.makedirs(save_dir, exist_ok=True)

    # Select hardware accelerator
    if device_name is not None:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"===========================================================")
    print(f" DPGNET TRAINING PIPELINE")
    print(f" Device: {device} | Target Epochs: {epochs} | Batch Size: {batch_size}")
    print(f" Early Stopping Patience: {patience} | Min Delta: {min_delta}")
    print(f" Checkpoint Directory: {save_dir}")
    print(f"===========================================================")

    train_loader, val_loader, dataset = get_dataloaders(
        data_root=data_root, batch_size=batch_size
    )

    model = DPGNet(pth_path=None).to(device)
    criterion = DPGNetLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_dice = 0.0
    start_epoch = 1
    patience_counter = 0
    history = {
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "lr": [],
    }

    # Resume from checkpoint if requested
    latest_path = os.path.join(save_dir, "latest_model.pth")
    if resume and os.path.isfile(latest_path):
        chk = torch.load(latest_path, map_location=device)
        model.load_state_dict(chk["model_state"])
        optimizer.load_state_dict(chk["optimizer_state"])
        start_epoch = chk["epoch"] + 1
        history = chk.get("history", history)
        best_path = os.path.join(save_dir, "best_model.pth")
        if os.path.isfile(best_path):
            best_chk = torch.load(best_path, map_location=device)
            best_val_dice = best_chk.get("val_dice", 0.0)
        print(f" Resumed training from epoch {start_epoch} (Best Val Dice: {best_val_dice:.4f})")

    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["train_dice"].append(round(train_dice, 4))
        history["train_iou"].append(round(train_iou, 4))
        history["lr"].append(current_lr)

        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        improved = False

        if epoch % val_interval == 0:
            val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(round(val_loss, 4))
            history["val_dice"].append(round(val_dice, 4))
            history["val_iou"].append(round(val_iou, 4))

            # Check for improvement
            if val_dice > best_val_dice + min_delta:
                improved = True
                patience_counter = 0
                best_val_dice = val_dice
                best_path = os.path.join(save_dir, "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_dice": val_dice,
                        "val_iou": val_iou,
                    },
                    best_path,
                )
                print(f" -> Saved new best model (Val Dice: {val_dice:.4f}) to {best_path}")
            else:
                patience_counter += 1

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
            },
            latest_path,
        )

        # Save history log
        history_path = os.path.join(save_dir, "train_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | LR: {current_lr:.6f}"
        )

        # Early Stopping Check 1: Patience exceeded (convergence)
        if patience_counter >= patience:
            print(f"\n[EARLY STOP] Validation Dice did not improve for {patience} epochs.")
            print(f"Stopping early to preserve best weights.")
            break

        # Early Stopping Check 2: Overfitting signs (loss diverging while train drops)
        if len(history["val_loss"]) >= 3:
            v_losses = history["val_loss"]
            t_losses = history["train_loss"]
            if (v_losses[-1] > v_losses[-2] > v_losses[-3]) and (t_losses[-1] < t_losses[-3] - 0.2):
                print(f"\n[EARLY STOP] Overfitting detected: validation loss increasing continuously.")
                print(f"Stopping early to prevent overfitting.")
                break

    total_time = time.time() - start_time
    print(f"===========================================================")
    print(f" Training session ended in {total_time / 60:.1f} minutes.")
    print(f" Best Validation Dice: {best_val_dice:.4f}")
    print(f" Checkpoints saved to: {save_dir}")
    print(f"===========================================================")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train DPGNet on Microglia Dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--save-dir", type=str, default="dpgnet/output/checkpoints", help="Save directory"
    )
    parser.add_argument(
        "--data-root", type=str, default="contour/output", help="Data root folder"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (mps, cuda, cpu)")
    parser.add_argument("--val-interval", type=int, default=1, help="Validation interval")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=0.001, help="Min delta for early stop")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    args = parser.parse_args()
    train_dpgnet(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        data_root=args.data_root,
        device_name=args.device,
        val_interval=args.val_interval,
        patience=args.patience,
        min_delta=args.min_delta,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
