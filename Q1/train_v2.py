import os
import torch
import argparse
import random
import numpy as np

from config import Config
from data_loader_v2 import get_data_loaders, analyze_class_distribution
from models.yolo11_classifier import YOLO11Classifier
from models.resnet_baseline import ResNet18Baseline
from trainer import Trainer


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(model_type, num_classes, pretrained=True):
    if model_type == 'yolo11x-cls':
        return YOLO11Classifier(num_classes, pretrained)
    elif model_type == 'resnet18':
        return ResNet18Baseline(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main():
    parser = argparse.ArgumentParser(description='Train Crop Disease Recognition Model')
    parser.add_argument('--model_type', type=str, default='yolo11x-cls', choices=['yolo11x-cls', 'resnet18'],
                        help='Model architecture')
    parser.add_argument('--train_dir', type=str, default=Config.DATA_DIR,
                        help='Training data directory')
    parser.add_argument('--val_dir', type=str, default=Config.VAL_DIR,
                        help='Validation data directory')
    parser.add_argument('--save_dir', type=str, default=Config.SAVE_DIR,
                        help='Directory to save model checkpoints')
    parser.add_argument('--figures_dir', type=str, default=Config.FIGURES_DIR,
                        help='Directory to save figures')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=Config.LR,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=Config.WEIGHT_DECAY,
                        help='Weight decay')
    parser.add_argument('--label_smoothing', type=float, default=Config.LABEL_SMOOTHING,
                        help='Label smoothing factor')
    parser.add_argument('--use_amp', action='store_true', default=Config.USE_AMP,
                        help='Use automatic mixed precision')
    parser.add_argument('--use_ema', action='store_true', default=Config.USE_EMA,
                        help='Use exponential moving average')
    parser.add_argument('--mixup_alpha', type=float, default=Config.MIXUP_ALPHA,
                        help='Mixup alpha parameter')
    parser.add_argument('--cutmix_alpha', type=float, default=Config.CUTMIX_ALPHA,
                        help='CutMix alpha parameter')
    parser.add_argument('--mixup_prob', type=float, default=Config.MIXUP_PROB,
                        help='Probability of applying Mixup/CutMix')
    parser.add_argument('--seed', type=int, default=Config.SEED,
                        help='Random seed')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = Config()
    config.update(
        MODEL_TYPE=args.model_type,
        DATA_DIR=args.train_dir,
        VAL_DIR=args.val_dir,
        SAVE_DIR=args.save_dir,
        FIGURES_DIR=args.figures_dir,
        BATCH_SIZE=args.batch_size,
        EPOCHS=args.epochs,
        LR=args.lr,
        WEIGHT_DECAY=args.weight_decay,
        LABEL_SMOOTHING=args.label_smoothing,
        USE_AMP=args.use_amp,
        USE_EMA=args.use_ema,
        MIXUP_ALPHA=args.mixup_alpha,
        CUTMIX_ALPHA=args.cutmix_alpha,
        MIXUP_PROB=args.mixup_prob,
        SEED=args.seed
    )
    
    print("=== Configuration ===")
    print(f"Model: {config.MODEL_TYPE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning rate: {config.LR}")
    print(f"Weight decay: {config.WEIGHT_DECAY}")
    print(f"Label smoothing: {config.LABEL_SMOOTHING}")
    print(f"Use AMP: {config.USE_AMP}")
    print(f"Use EMA: {config.USE_EMA}")
    print(f"Mixup alpha: {config.MIXUP_ALPHA}")
    print(f"CutMix alpha: {config.CUTMIX_ALPHA}")
    print(f"Mixup probability: {config.MIXUP_PROB}")
    print(f"Image size: {config.IMG_SIZE}")
    print(f"Random seed: {config.SEED}")
    print("====================")
    
    print("Analyzing dataset...")
    class_dist = analyze_class_distribution(config.DATA_DIR)
    config.NUM_CLASSES = class_dist['num_classes']
    print(f"Found {config.NUM_CLASSES} classes")
    print(f"Total samples: {class_dist['total_samples']}")
    print(f"Min samples per class: {class_dist['min_samples']}")
    print(f"Max samples per class: {class_dist['max_samples']}")
    
    print("Creating data loaders...")
    train_loader, val_loader, train_size, val_size = get_data_loaders(config)
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    print("Creating model...")
    model = create_model(config.MODEL_TYPE, config.NUM_CLASSES, config.PRETRAINED)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.1f} GB")
    
    trainer = Trainer(model, train_loader, val_loader, config)
    
    print("Starting training...")
    metrics_history = trainer.train()
    
    print("Training completed!")
    print(f"Best F1: {trainer.best_f1:.4f}")
    print(f"Best Acc: {trainer.best_acc:.4f}")
    print(f"Results saved to: {config.SAVE_DIR}")
    print(f"Figures saved to: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()