import os
import math
import json
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import Config
from data_loader import get_data_loaders
from model import create_model, get_model_info
from utils import save_checkpoint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def smooth_one_hot(targets, num_classes, smoothing):
    off = smoothing / num_classes
    on = 1.0 - smoothing + off
    y = torch.full((targets.size(0), num_classes), off, device=targets.device)
    y.scatter_(1, targets.unsqueeze(1), on)
    return y


def rand_bbox(size, lam):
    W = size[3]
    H = size[2]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mix_batch(x, y, cfg, num_classes):
    x = x.clone()
    if torch.rand(1).item() > cfg.MIXUP_PROB:
        targets = smooth_one_hot(y, num_classes, cfg.LABEL_SMOOTHING)
        return x, targets
    index = torch.randperm(x.size(0), device=x.device)
    use_mixup = torch.rand(1).item() < 0.5
    if use_mixup:
        lam = np.random.beta(cfg.MIXUP_ALPHA, cfg.MIXUP_ALPHA)
        x = lam * x + (1 - lam) * x[index]
    else:
        lam = np.random.beta(cfg.CUTMIX_ALPHA, cfg.CUTMIX_ALPHA)
        x1, y1, x2, y2 = rand_bbox(x.size(), lam)
        x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1)) / (x.size(2) * x.size(3))
    target_a = smooth_one_hot(y, num_classes, cfg.LABEL_SMOOTHING)
    target_b = smooth_one_hot(y[index], num_classes, cfg.LABEL_SMOOTHING)
    targets = lam * target_a + (1 - lam) * target_b
    return x, targets


def soft_ce(logits, targets):
    log_probs = torch.log_softmax(logits, dim=1)
    loss = -(targets * log_probs).sum(dim=1)
    return loss.mean()


def compute_lr(epoch, cfg):
    if cfg.WARMUP_EPOCHS and epoch < cfg.WARMUP_EPOCHS:
        if cfg.WARMUP_EPOCHS == 1:
            return cfg.LR
        alpha = epoch / (cfg.WARMUP_EPOCHS - 1)
        return cfg.WARMUP_START_LR + (cfg.LR - cfg.WARMUP_START_LR) * alpha
    progress = (epoch - cfg.WARMUP_EPOCHS) / max(1, cfg.EPOCHS - cfg.WARMUP_EPOCHS)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.MIN_LR + (cfg.LR - cfg.MIN_LR) * cosine


def set_lr(optimizer, value):
    for group in optimizer.param_groups:
        group['lr'] = value


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state.items()}


def train_epoch(model, loader, optimizer, scaler, device, cfg, num_classes, use_amp, ema):
    model.train()
    total_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    samples = 0
    preds_all = []
    labels_all = []
    for x, y in tqdm(loader, desc='Train', leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mixed_x, soft_targets = mix_batch(x, y, cfg, num_classes)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(mixed_x)
            loss = soft_ce(outputs, soft_targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema:
            ema.update(model)
        total_loss += loss.item() * x.size(0)
        preds = torch.topk(outputs.detach(), k=3, dim=1).indices
        total_top1 += (preds[:, 0] == y).sum().item()
        total_top3 += (preds == y.unsqueeze(1)).any(dim=1).sum().item()
        samples += x.size(0)
        preds_all.append(preds[:, 0].cpu())
        labels_all.append(y.cpu())
    preds_cat = torch.cat(preds_all).numpy()
    labels_cat = torch.cat(labels_all).numpy()
    precision = precision_score(labels_cat, preds_cat, average='macro', zero_division=0)
    recall = recall_score(labels_cat, preds_cat, average='macro', zero_division=0)
    f1 = f1_score(labels_cat, preds_cat, average='macro', zero_division=0)
    return {
        'loss': total_loss / samples,
        'top1': total_top1 / samples,
        'top3': total_top3 / samples,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def eval_epoch(model, loader, device, num_classes, use_amp, collect=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    samples = 0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc='Val', leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.topk(outputs, k=3, dim=1).indices
            total_top1 += (preds[:, 0] == y).sum().item()
            total_top3 += (preds == y.unsqueeze(1)).any(dim=1).sum().item()
            samples += x.size(0)
            preds_all.append(preds[:, 0].cpu())
            labels_all.append(y.cpu())
    preds_cat = torch.cat(preds_all).numpy()
    labels_cat = torch.cat(labels_all).numpy()
    precision = precision_score(labels_cat, preds_cat, average='macro', zero_division=0)
    recall = recall_score(labels_cat, preds_cat, average='macro', zero_division=0)
    f1 = f1_score(labels_cat, preds_cat, average='macro', zero_division=0)
    if collect:
        cm = confusion_matrix(labels_cat, preds_cat)
        report = classification_report(labels_cat, preds_cat, output_dict=True)
    else:
        cm = None
        report = None
    return {
        'loss': total_loss / samples,
        'top1': total_top1 / samples,
        'top3': total_top3 / samples,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm,
        'report': report
    }


def plot_curves(history, save_dir):
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', color='#ff7f0e')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(epochs, history['train_top1'], label='Train Top-1', color='#2ca02c')
    axes[1].plot(epochs, history['val_top1'], label='Val Top-1', color='#d62728')
    axes[1].plot(epochs, history['val_top3'], label='Val Top-3', color='#9467bd')
    axes[1].set_title('Accuracy Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[2].plot(epochs, history['train_f1'], label='Train F1', color='#8c564b')
    axes[2].plot(epochs, history['val_f1'], label='Val F1', color='#ff9896')
    axes[2].set_title('F1 and LR')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1')
    ax2_lr = axes[2].twinx()
    ax2_lr.plot(epochs, history['lr'], label='LR', color='#17becf', linestyle='--')
    ax2_lr.set_ylabel('LR')
    lines, labels = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2_lr.get_legend_handles_labels()
    axes[2].legend(lines + lines2, labels + labels2, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_cm(cm, save_dir):
    plt.style.use('seaborn-v0_8-paper')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap='viridis', cbar=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def run(cfg, train_dir, val_dir):
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    set_seed(cfg.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')
    train_loader, val_loader, _, _ = get_data_loaders(
        train_dir, val_dir, batch_size=cfg.BATCH_SIZE, num_workers=8
    )
    labels = [label for _, label in train_loader.dataset.samples]
    num_classes = len(set(labels))
    model = create_model(cfg.MODEL_TYPE, num_classes, cfg.PRETRAINED).to(device)
    info = get_model_info(model)
    print(f"Model params: {info['total_parameters']/1e6:.2f}M")
    optimizer = AdamW(model.parameters(), lr=cfg.WARMUP_START_LR, weight_decay=cfg.WEIGHT_DECAY)
    use_amp = cfg.USE_AMP and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    ema = EMA(model, cfg.EMA_DECAY) if cfg.USE_EMA else None
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_top1': [],
        'val_top1': [],
        'val_top3': [],
        'train_f1': [],
        'val_f1': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
        'lr': []
    }
    best_f1 = 0.0
    best_loss = float('inf')
    start_time = time.time()
    for epoch in range(cfg.EPOCHS):
        lr = compute_lr(epoch, cfg)
        set_lr(optimizer, lr)
        print(f'Epoch {epoch + 1}/{cfg.EPOCHS} | lr {lr:.6f}')
        train_stats = train_epoch(model, train_loader, optimizer, scaler, device, cfg, num_classes, use_amp, ema)
        val_stats = eval_epoch(model, val_loader, device, num_classes, use_amp, collect=False)
        history['train_loss'].append(train_stats['loss'])
        history['val_loss'].append(val_stats['loss'])
        history['train_top1'].append(train_stats['top1'])
        history['val_top1'].append(val_stats['top1'])
        history['val_top3'].append(val_stats['top3'])
        history['train_f1'].append(train_stats['f1'])
        history['val_f1'].append(val_stats['f1'])
        history['train_precision'].append(train_stats['precision'])
        history['val_precision'].append(val_stats['precision'])
        history['train_recall'].append(train_stats['recall'])
        history['val_recall'].append(val_stats['recall'])
        history['lr'].append(lr)
        print(
            f"Train Loss {train_stats['loss']:.4f} | "
            f"Acc {train_stats['top1']:.4f} | "
            f"Prec {train_stats['precision']:.4f} | "
            f"Rec {train_stats['recall']:.4f} | "
            f"F1 {train_stats['f1']:.4f}"
        )
        print(
            f"Val   Loss {val_stats['loss']:.4f} | "
            f"Acc {val_stats['top1']:.4f} | "
            f"Top-3 {val_stats['top3']:.4f} | "
            f"Prec {val_stats['precision']:.4f} | "
            f"Rec {val_stats['recall']:.4f} | "
            f"F1 {val_stats['f1']:.4f}"
        )
        if val_stats['f1'] > best_f1:
            best_f1 = val_stats['f1']
            ckpt_path = os.path.join(cfg.SAVE_DIR, 'best_model_val_f1.pth')
            save_checkpoint({'model': model.state_dict(), 'ema': ema.state_dict() if ema else None}, ckpt_path)
        if val_stats['loss'] < best_loss:
            best_loss = val_stats['loss']
            ckpt_path = os.path.join(cfg.SAVE_DIR, 'best_model_val_loss.pth')
            save_checkpoint({'model': model.state_dict(), 'ema': ema.state_dict() if ema else None}, ckpt_path)
        latest_path = os.path.join(cfg.SAVE_DIR, 'latest_model.pth')
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if use_amp else None,
            'ema': ema.state_dict() if ema else None,
            'history': history
        }, latest_path)
    if ema:
        ema.apply(model)
    final_stats = eval_epoch(model, val_loader, device, num_classes, use_amp, collect=True)
    if ema:
        ema.restore(model)
    duration = time.time() - start_time
    plot_curves(history, cfg.FIGURES_DIR)
    plot_cm(final_stats['cm'], cfg.FIGURES_DIR)
    report = {
        'duration_min': duration / 60,
        'best_f1': best_f1,
        'best_loss': best_loss,
        'final_top1': final_stats['top1'],
        'final_top3': final_stats['top3'],
        'final_f1': final_stats['f1'],
        'per_class': final_stats['report']
    }
    with open(os.path.join(cfg.SAVE_DIR, 'training_summary.json'), 'w') as f:
        json.dump({'history': history, 'report': report}, f, indent=2)
    print(f"Training finished in {duration/60:.1f} min. Final F1 {final_stats['f1']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default=Config.DATA_DIR)
    parser.add_argument('--val_dir', type=str, default=Config.VAL_DIR)
    parser.add_argument('--model', type=str, default=Config.MODEL_TYPE)
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=Config.LR)
    return parser.parse_args()


def main():
    args = parse_args()
    Config.MODEL_TYPE = args.model
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LR = args.lr
    run(Config, args.train_dir, args.val_dir)


if __name__ == '__main__':
    main()
