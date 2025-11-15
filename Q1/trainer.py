import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import copy
import json
import os
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, top_k_accuracy_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from transforms.mixup import MixupCutMix, SoftTargetCrossEntropy
from visualizer import plot_training_curves, plot_confusion_matrix, plot_per_class_metrics


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1. - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        self.soft_criterion = SoftTargetCrossEntropy()
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.LR, 
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=config.EPOCHS - config.WARMUP_EPOCHS,
            eta_min=config.MIN_LR
        )
        
        self.mixup_cutmix = MixupCutMix(
            mixup_alpha=config.MIXUP_ALPHA,
            cutmix_alpha=config.CUTMIX_ALPHA,
            mixup_prob=config.MIXUP_PROB
        )
        
        self.ema = EMA(self.model, decay=config.EMA_DECAY) if config.USE_EMA else None
        self.scaler = GradScaler() if config.USE_AMP else None
        
        self.best_f1 = 0.0
        self.best_acc = 0.0
        self.metrics_history = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': [],
            'train_precisions_macro': [],
            'val_precisions_macro': [],
            'train_recalls_macro': [],
            'val_recalls_macro': [],
            'train_f1_macros': [],
            'val_f1_macros': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        os.makedirs(config.FIGURES_DIR, exist_ok=True)

    def warmup_lr(self, current_epoch):
        if current_epoch < self.config.WARMUP_EPOCHS:
            lr = self.config.WARMUP_START_LR + (self.config.LR - self.config.WARMUP_START_LR) * (
                current_epoch / self.config.WARMUP_EPOCHS
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        return self.config.LR

    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        top1_acc = np.mean(y_true == y_pred)
        
        if y_prob is not None:
            top3_acc = top_k_accuracy_score(y_true, y_prob, k=3)
        else:
            top3_acc = 0.0
            
        return {
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'top1_acc': top1_acc,
            'top3_acc': top3_acc
        }

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        for inputs, labels in tqdm(self.train_loader, desc=f'Epoch {epoch+1} Training'):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            if np.random.random() < self.config.MIXUP_PROB:
                inputs, y_a, y_b, lam = self.mixup_cutmix((inputs, labels))
                
                self.optimizer.zero_grad()
                
                if self.config.USE_AMP:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.soft_criterion(outputs, y_a) * lam + self.soft_criterion(outputs, y_b) * (1 - lam)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.soft_criterion(outputs, y_a) * lam + self.soft_criterion(outputs, y_b) * (1 - lam)
                    
                    loss.backward()
                    self.optimizer.step()
            else:
                self.optimizer.zero_grad()
                
                if self.config.USE_AMP:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    loss.backward()
                    self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                probs = torch.softmax(outputs, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        if self.ema:
            self.ema.update()
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_metrics = self.calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        return epoch_loss, epoch_metrics

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if self.config.USE_AMP:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                probs = torch.softmax(outputs, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_metrics = self.calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        return epoch_loss, epoch_metrics, np.array(all_labels), np.array(all_preds)

    def train(self):
        print(f"Starting training for {self.config.EPOCHS} epochs")
        print(f"Using device: {self.device}")
        print(f"Using AMP: {self.config.USE_AMP}")
        print(f"Using EMA: {self.config.USE_EMA}")
        
        start_time = time.time()
        
        for epoch in range(self.config.EPOCHS):
            epoch_start_time = time.time()
            
            current_lr = self.warmup_lr(epoch)
            
            train_loss, train_metrics = self.train_epoch(epoch)
            
            if epoch >= self.config.WARMUP_EPOCHS:
                self.scheduler.step()
            
            if self.ema:
                self.ema.apply_shadow()
                val_loss, val_metrics, val_labels, val_preds = self.validate_epoch()
                self.ema.restore()
            else:
                val_loss, val_metrics, val_labels, val_preds = self.validate_epoch()
            
            epoch_time = time.time() - epoch_start_time
            
            self.metrics_history['train_losses'].append(train_loss)
            self.metrics_history['val_losses'].append(val_loss)
            self.metrics_history['train_accs'].append(train_metrics['top1_acc'])
            self.metrics_history['val_accs'].append(val_metrics['top1_acc'])
            self.metrics_history['train_precisions_macro'].append(train_metrics['precision_macro'])
            self.metrics_history['val_precisions_macro'].append(val_metrics['precision_macro'])
            self.metrics_history['train_recalls_macro'].append(train_metrics['recall_macro'])
            self.metrics_history['val_recalls_macro'].append(val_metrics['recall_macro'])
            self.metrics_history['train_f1_macros'].append(train_metrics['f1_macro'])
            self.metrics_history['val_f1_macros'].append(val_metrics['f1_macro'])
            self.metrics_history['learning_rates'].append(current_lr)
            self.metrics_history['epoch_times'].append(epoch_time)
            
            print(f'Epoch {epoch+1}/{self.config.EPOCHS}')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Train Acc: {train_metrics["top1_acc"]:.4f}, Val Acc: {val_metrics["top1_acc"]:.4f}')
            print(f'Train F1: {train_metrics["f1_macro"]:.4f}, Val F1: {val_metrics["f1_macro"]:.4f}')
            print(f'Top-3 Acc: {val_metrics["top3_acc"]:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s')
            
            if val_metrics['f1_macro'] > self.best_f1:
                self.best_f1 = val_metrics['f1_macro']
                self.best_acc = val_metrics['top1_acc']
                
                if self.ema:
                    self.ema.apply_shadow()
                    torch.save(self.model.state_dict(), os.path.join(self.config.SAVE_DIR, 'best_model_f1.pth'))
                    self.ema.restore()
                else:
                    torch.save(self.model.state_dict(), os.path.join(self.config.SAVE_DIR, 'best_model_f1.pth'))
                
                print(f"New best F1: {self.best_f1:.4f}")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        print(f"Best F1: {self.best_f1:.4f}, Best Acc: {self.best_acc:.4f}")
        
        self.save_metrics()
        self.plot_results(val_labels, val_preds)
        
        return self.metrics_history

    def save_metrics(self, total_time):
        final_metrics = {
            'best_f1': float(self.best_f1),
            'best_acc': float(self.best_acc),
            'total_time': float(total_time),
            'config': {
                'model_type': self.config.MODEL_TYPE,
                'batch_size': self.config.BATCH_SIZE,
                'epochs': self.config.EPOCHS,
                'lr': self.config.LR,
                'weight_decay': self.config.WEIGHT_DECAY,
                'label_smoothing': self.config.LABEL_SMOOTHING,
                'use_amp': self.config.USE_AMP,
                'use_ema': self.config.USE_EMA,
                'mixup_alpha': self.config.MIXUP_ALPHA,
                'cutmix_alpha': self.config.CUTMIX_ALPHA,
                'mixup_prob': self.config.MIXUP_PROB
            },
            'metrics_history': self.metrics_history
        }
        
        with open(os.path.join(self.config.SAVE_DIR, 'training_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2)

    def plot_results(self, val_labels, val_preds):
        plot_training_curves(self.metrics_history, self.config.FIGURES_DIR)
        plot_confusion_matrix(
            val_labels, val_preds, 
            self.config.NUM_CLASSES, 
            self.config.FIGURES_DIR
        )