import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import analyze_class_distribution

def plot_class_distribution(train_dir, val_dir=None, save_path=None):
    # 设置美观的绘图风格
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 获取训练集类别分布
    train_dist = analyze_class_distribution(train_dir)
    train_counts = train_dist['class_counts']
    
    # 如果有验证集，也获取其分布
    val_counts = {}
    if val_dir:
        val_dist = analyze_class_distribution(val_dir)
        val_counts = val_dist['class_counts']
    
    # 准备数据
    classes = sorted(train_counts.keys())
    train_values = [train_counts.get(c, 0) for c in classes]
    val_values = [val_counts.get(c, 0) for c in classes] if val_counts else None
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 第一个子图：条形图
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, train_values, width, label='Training Set', alpha=0.8,
                    color='skyblue', edgecolor='navy', linewidth=0.8)
    
    if val_values:
        bars2 = ax1.bar(x + width/2, val_values, width, label='Validation Set', alpha=0.8,
                        color='lightcoral', edgecolor='darkred', linewidth=0.8)
    
    ax1.set_xlabel('Class', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax1.set_title('Crop Disease Dataset Class Distribution', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Class {c}' for c in classes], rotation=45, ha='right')
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # 在条形图上添加数值标签
    for bars in [bars1]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    if val_values:
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax1.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    # 第二个子图：饼图显示训练集分布
    # 只显示前10个最多的类别，其余合并为"其他"
    sorted_counts = sorted(train_counts.items(), key=lambda x: x[1], reverse=True)
    top_10 = sorted_counts[:10]
    other_count = sum(count for _, count in sorted_counts[10:])
    
    if other_count > 0:
        pie_labels = [f'Class {c}' for c, _ in top_10] + ['Others']
        pie_values = [count for _, count in top_10] + [other_count]
    else:
        pie_labels = [f'Class {c}' for c, _ in top_10]
        pie_values = [count for _, count in top_10]
    
    # 创建饼图
    colors = plt.cm.Set3(np.linspace(0, 1, len(pie_labels)))
    wedges, texts, autotexts = ax2.pie(pie_values, labels=pie_labels, autopct='%1.1f%%',
                                       startangle=90, colors=colors, 
                                       textprops={'fontsize': 10})
    
    # 美化饼图文本
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax2.set_title('Training Set Class Distribution (Top 10 Classes)', fontsize=14, fontweight='bold')
    
    # 添加统计信息
    total_samples = sum(train_values)
    num_classes = len(classes)
    avg_samples = total_samples / num_classes
    
    stats_text = f'Total Samples: {total_samples:,}\n'
    stats_text += f'Number of Classes: {num_classes}\n'
    stats_text += f'Avg Samples per Class: {avg_samples:.1f}\n'
    stats_text += f'Max Samples: {max(train_values):,}\n'
    stats_text += f'Min Samples: {min(train_values):,}'
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")
    else:
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        print("Class distribution plot saved to: class_distribution.png")
    
    plt.show()
    
    return train_counts, val_counts if val_counts else None

def main():
    train_dir = '../AgriculturalDisease_trainingset'
    val_dir = '../AgriculturalDisease_validationset'
    
    print("Generating class distribution plot...")
    train_counts, val_counts = plot_class_distribution(train_dir, val_dir, 'class_distribution.png')
    
    print("\nClass Distribution Statistics:")
    print(f"Training Set Classes: {len(train_counts)}")
    if val_counts:
        print(f"Validation Set Classes: {len(val_counts)}")
    
    print("\nTop 10 Classes by Sample Count:")
    sorted_counts = sorted(train_counts.items(), key=lambda x: x[1], reverse=True)
    for class_id, count in sorted_counts[:10]:
        print(f"Class {class_id}: {count} samples")

if __name__ == "__main__":
    main()