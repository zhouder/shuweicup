import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-paper')


def set_paper_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': True,
        'legend.edgecolor': 'black'
    })


def plot_training_curves(metrics, save_dir):
    set_paper_style()
    
    epochs = range(1, len(metrics['train_losses']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training and Validation Metrics', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(epochs, metrics['train_losses'], 'b-', linewidth=2, label='Training Loss')
    axes[0, 0].plot(epochs, metrics['val_losses'], 'r-', linewidth=2, label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, metrics['train_accs'], 'b-', linewidth=2, label='Training Accuracy')
    axes[0, 1].plot(epochs, metrics['val_accs'], 'r-', linewidth=2, label='Validation Accuracy')
    axes[0, 1].set_title('Top-1 Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(epochs, metrics['train_f1_macros'], 'b-', linewidth=2, label='Training F1')
    axes[0, 2].plot(epochs, metrics['val_f1_macros'], 'r-', linewidth=2, label='Validation F1')
    axes[0, 2].set_title('Macro F1 Score')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, metrics['train_precisions_macro'], 'b-', linewidth=2, label='Training Precision')
    axes[1, 0].plot(epochs, metrics['val_precisions_macro'], 'r-', linewidth=2, label='Validation Precision')
    axes[1, 0].set_title('Macro Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, metrics['train_recalls_macro'], 'b-', linewidth=2, label='Training Recall')
    axes[1, 1].plot(epochs, metrics['val_recalls_macro'], 'r-', linewidth=2, label='Validation Recall')
    axes[1, 1].set_title('Macro Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(epochs, metrics['learning_rates'], 'g-', linewidth=2)
    axes[1, 2].set_title('Learning Rate Schedule')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.pdf'), format='pdf')
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), format='png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, num_classes, save_dir, class_names=None):
    set_paper_style()
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.pdf'), format='pdf')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), format='png')
    plt.close()


def plot_per_class_metrics(report, save_dir, class_names=None):
    set_paper_style()
    
    if class_names is None:
        class_names = list(report.keys())[:-3]
    
    precision = [report[class_name]['precision'] for class_name in class_names]
    recall = [report[class_name]['recall'] for class_name in class_names]
    f1_score = [report[class_name]['f1-score'] for class_name in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax.bar(x - width, precision, width, label='Precision', color='#1f77b4', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', color='#ff7f0e', alpha=0.8)
    ax.bar(x + width, f1_score, width, label='F1-Score', color='#2ca02c', alpha=0.8)
    
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_metrics.pdf'), format='pdf')
    plt.savefig(os.path.join(save_dir, 'per_class_metrics.png'), format='png')
    plt.close()


def plot_model_comparison(results_dict, save_dir):
    set_paper_style()
    
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        values = [results_dict[model][metric] for model in models]
        
        bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_ylabel('Score')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.pdf'), format='pdf')
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), format='png')
    plt.close()


def plot_learning_rate_schedule(lr_schedule, save_dir, total_epochs):
    set_paper_style()
    
    epochs = range(1, total_epochs + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lr_schedule, 'b-', linewidth=2)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lr_schedule.pdf'), format='pdf')
    plt.savefig(os.path.join(save_dir, 'lr_schedule.png'), format='png')
    plt.close()