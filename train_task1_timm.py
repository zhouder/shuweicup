import os
import random
import time
import csv
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

import timm
from tqdm.auto import tqdm


# ================== 基本配置 ==================
# 当前文件：Problem B/Q1/train_task1_timm.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_IMAGES = os.path.join(PROJECT_ROOT, "AgriculturalDisease_trainingset", "images")
VAL_IMAGES   = os.path.join(PROJECT_ROOT, "AgriculturalDisease_validationset", "images")

NUM_CLASSES = 61          # 问题一：原始 61 类
IMAGE_SIZE = 320
BATCH_SIZE = 64
NUM_EPOCHS = 80
SEED = 2025
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_DIR = "checkpoints(zhouder)"
BEST_CKPT = os.path.join(CKPT_DIR, "task1_best_timm.pth")
LAST_CKPT = os.path.join(CKPT_DIR, "task1_last_timm.pth")
METRICS_CSV = os.path.join(CKPT_DIR, "task1_metrics.csv")

RESUME_TRAINING = True   # 断点续训开关


# ================== 工具函数 ==================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image_file(fname: str) -> bool:
    ext = os.path.splitext(fname)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp"]


def parse_label_from_name(fname: str) -> int:
    """
    文件名：23_17842.jpg -> 标签 23
    """
    base = os.path.basename(fname)
    prefix = base.split("_")[0]
    return int(prefix)


def compute_class_weights(labels, num_classes: int):
    counter = Counter(labels)
    counts = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        counts[c] = counter.get(c, 0)
    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-6)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ================== Dataset ==================
class FileNameDataset(Dataset):
    """
    问题一：直接用文件名前缀作为 61 类标签
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        for fname in os.listdir(root):
            if not is_image_file(fname):
                continue
            fpath = os.path.join(root, fname)
            try:
                label = parse_label_from_name(fname)
            except ValueError:
                print(f"[WARN] Skip file without numeric label: {fname}")
                continue
            self.samples.append((fpath, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found in {root}")

        self.labels = [s[1] for s in self.samples]
        print(f"[INFO] Loaded {len(self.samples)} images from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ================== 数据增强 ==================
def build_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.1)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


# ================== 指标计算 ==================
def compute_metrics_from_confmat(conf_mat: np.ndarray):
    """
    从混淆矩阵计算：
    - top1 accuracy
    - macro precision / recall / F1
    （注意：top3 单独在循环里算，不依赖混淆矩阵）
    """
    tp = np.diag(conf_mat).astype(np.float64)
    support = conf_mat.sum(axis=1).astype(np.float64)
    pred_count = conf_mat.sum(axis=0).astype(np.float64)

    eps = 1e-12
    precision = tp / (pred_count + eps)
    recall    = tp / (support + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    acc = tp.sum() / (conf_mat.sum() + eps)
    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f1 = f1.mean()

    metrics = {
        "acc": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
    }
    return metrics


# ================== 训练 & 验证 ==================
def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    running_loss = 0.0
    total = 0
    correct_top1 = 0
    correct_top3 = 0

    start_time = time.time()
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", ncols=100)

    for images, labels in pbar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        # top1 & top3
        _, top1 = outputs.topk(1, dim=1)
        _, top3 = outputs.topk(3, dim=1)
        correct_top1 += top1.eq(labels.view(-1, 1)).sum().item()
        correct_top3 += top3.eq(labels.view(-1, 1)).any(dim=1).sum().item()

        preds_np = top1.squeeze(1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        for t, p in zip(labels_np, preds_np):
            conf_mat[t, p] += 1

        avg_loss = running_loss / total
        acc1 = correct_top1 / total
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc1": f"{acc1:.4f}"})

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / total
    metrics = compute_metrics_from_confmat(conf_mat)
    metrics["acc1"] = correct_top1 / total
    metrics["acc3"] = correct_top3 / total

    print(
        f"[Epoch {epoch}] Train samples: {total}, "
        f"Time: {epoch_time:.1f}s, "
        f"Loss: {epoch_loss:.4f}, "
        f"Acc@1: {metrics['acc1']:.4f}, Acc@3: {metrics['acc3']:.4f}, "
        f"MacroF1: {metrics['macro_f1']:.4f}"
    )
    return epoch_loss, metrics, epoch_time


@torch.no_grad()
def validate(model, loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    total = 0
    correct_top1 = 0
    correct_top3 = 0

    start_time = time.time()
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    pbar = tqdm(loader, desc=f"Valid Epoch {epoch}", ncols=100)

    for images, labels in pbar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        _, top1 = outputs.topk(1, dim=1)
        _, top3 = outputs.topk(3, dim=1)
        correct_top1 += top1.eq(labels.view(-1, 1)).sum().item()
        correct_top3 += top3.eq(labels.view(-1, 1)).any(dim=1).sum().item()

        preds_np = top1.squeeze(1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        for t, p in zip(labels_np, preds_np):
            conf_mat[t, p] += 1

        avg_loss = running_loss / total
        acc1 = correct_top1 / total
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc1": f"{acc1:.4f}"})

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / total
    metrics = compute_metrics_from_confmat(conf_mat)
    metrics["acc1"] = correct_top1 / total
    metrics["acc3"] = correct_top3 / total

    print(
        f"[Epoch {epoch}] Val   samples: {total}, "
        f"Time: {epoch_time:.1f}s, "
        f"Loss: {epoch_loss:.4f}, "
        f"Acc@1: {metrics['acc1']:.4f}, Acc@3: {metrics['acc3']:.4f}, "
        f"MacroF1: {metrics['macro_f1']:.4f}"
    )
    return epoch_loss, metrics, epoch_time


# ================== CSV 记录 ==================
def init_metrics_csv(path):
    if os.path.exists(path):
        return
    header = [
        "epoch", "split", "loss",
        "acc1", "acc3",
        "macro_precision", "macro_recall", "macro_f1",
        "time_sec",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_metrics_csv(path, epoch, split, loss, metrics, time_sec):
    row = [
        epoch,
        split,
        f"{loss:.6f}",
        f"{metrics['acc1']:.6f}",
        f"{metrics['acc3']:.6f}",
        f"{metrics['macro_precision']:.6f}",
        f"{metrics['macro_recall']:.6f}",
        f"{metrics['macro_f1']:.6f}",
        f"{time_sec:.3f}",
    ]
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ================== Checkpoint 相关 ==================
def save_checkpoint(epoch, model, optimizer, scheduler, scaler,
                    best_acc1, is_best):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_acc1": best_acc1,
    }
    torch.save(state, LAST_CKPT)
    if is_best:
        torch.save(state, BEST_CKPT)
        print(f"[INFO] Best checkpoint updated at epoch {epoch}, acc1={best_acc1:.4f}")


def maybe_load_checkpoint(model, optimizer, scheduler, scaler):
    if RESUME_TRAINING and os.path.exists(LAST_CKPT):
        print(f"[INFO] Found existing checkpoint: {LAST_CKPT}, trying to resume...")
        state = torch.load(LAST_CKPT, map_location=DEVICE)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        start_epoch = state["epoch"] + 1
        best_acc1 = state.get("best_acc1", 0.0)
        print(f"[INFO] Resume from epoch {state['epoch']}, best_acc1={best_acc1:.4f}")
        return start_epoch, best_acc1
    else:
        return 1, 0.0


# ================== 主函数 ==================
def main():
    set_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)
    init_metrics_csv(METRICS_CSV)

    train_tf, val_tf = build_transforms()

    train_dataset = FileNameDataset(TRAIN_IMAGES, transform=train_tf)
    val_dataset   = FileNameDataset(VAL_IMAGES,   transform=val_tf)

    print(f"[INFO] Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    print(f"[DEBUG] Train label distribution: {Counter(train_dataset.labels)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    class_weights = compute_class_weights(train_dataset.labels, NUM_CLASSES).to(DEVICE)
    print("[INFO] Class weights computed (not yet used in loss).")

    # ----- 构建模型，并尝试在线加载预训练权重 -----
    try:
        model = timm.create_model(
            "convnext_tiny",
            pretrained=True,          # 这里会通过 HF / 镜像自动下载权重
            num_classes=NUM_CLASSES,
        )
        print("[INFO] Loaded ConvNeXt-Tiny with ImageNet pretrained weights.")
    except Exception as e:
        print("[WARN] Failed to load pretrained weights (may have no Internet / hub error). "
              "Falling back to random init.\n", str(e))
        model = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=NUM_CLASSES,
        )

    model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {n_params / 1e6:.2f} M")

    criterion = nn.CrossEntropyLoss(
        # 需要的话可以改成 weight=class_weights
        label_smoothing=0.1
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.05
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    scaler = GradScaler()

    start_epoch, best_acc1 = maybe_load_checkpoint(
        model, optimizer, scheduler, scaler
    )

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_loss, train_metrics, train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch
        )
        val_loss, val_metrics, val_time = validate(
            model, val_loader, criterion, epoch
        )
        scheduler.step()

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} Acc@1: {train_metrics['acc1']:.4f} "
            f"Acc@3: {train_metrics['acc3']:.4f} MacroF1: {train_metrics['macro_f1']:.4f} ({train_time:.1f}s) | "
            f"Val Loss: {val_loss:.4f} Acc@1: {val_metrics['acc1']:.4f} "
            f"Acc@3: {val_metrics['acc3']:.4f} MacroF1: {val_metrics['macro_f1']:.4f} ({val_time:.1f}s)"
        )

        append_metrics_csv(METRICS_CSV, epoch, "train", train_loss, train_metrics, train_time)
        append_metrics_csv(METRICS_CSV, epoch, "val",   val_loss,  val_metrics,  val_time)

        # 以 Val Acc@1 作为选最优模型的标准
        is_best = val_metrics["acc1"] > best_acc1
        if is_best:
            best_acc1 = val_metrics["acc1"]
        save_checkpoint(epoch, model, optimizer, scheduler, scaler,
                        best_acc1, is_best)

    print("Training finished. Best Val Acc@1:", best_acc1)


if __name__ == "__main__":
    main()
