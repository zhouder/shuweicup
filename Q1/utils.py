import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pathlib import Path

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename, map_location=None):
    if not os.path.exists(filename):
        print(f"No checkpoint found at {filename}")
        return {}
    
    if map_location is None:
        map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(filename, map_location=map_location)
        return checkpoint
    except RuntimeError as e:
        print(f"Error loading checkpoint {filename}: {e}")
        print("This might be due to a corrupted model file.")
        return {}
    except Exception as e:
        print(f"Unexpected error loading checkpoint {filename}: {e}")
        return {}

def calculate_metrics(y_true, y_pred, class_names=None):
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(max(max(y_true), max(y_pred)) + 1)]
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    return {
        'confusion_matrix': cm,
        'classification_report': report
    }

def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_class_mapping(data_dir, output_file='class_mapping.json'):
    class_mapping = {}
    
    train_images_dir = os.path.join(data_dir, 'images')
    if os.path.exists(train_images_dir):
        for filename in os.listdir(train_images_dir):
            if os.path.splitext(filename)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
                try:
                    class_id = int(filename.split('_')[0])
                    if class_id not in class_mapping:
                        class_mapping[class_id] = f"Disease_{class_id}"
                except (ValueError, IndexError):
                    print(f"Warning: Could not extract class from filename: {filename}")
    
    val_images_dir = os.path.join(data_dir, 'images')
    if os.path.exists(val_images_dir) and val_images_dir != train_images_dir:
        for filename in os.listdir(val_images_dir):
            if os.path.splitext(filename)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
                try:
                    class_id = int(filename.split('_')[0])
                    if class_id not in class_mapping:
                        class_mapping[class_id] = f"Disease_{class_id}"
                except (ValueError, IndexError):
                    print(f"Warning: Could not extract class from filename: {filename}")
    
    sorted_mapping = {str(k): v for k, v in sorted(class_mapping.items())}
    
    with open(output_file, 'w') as f:
        json.dump(sorted_mapping, f, indent=2)
    
    print(f"Class mapping saved to {output_file}")
    return sorted_mapping

def get_image_statistics(data_dir):
    from PIL import Image
    import numpy as np
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = []
    
    images_dir = os.path.join(data_dir, 'images')
    if os.path.exists(images_dir):
        for root, _, files in os.walk(images_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print("No images found")
        return {}
    
    widths = []
    heights = []
    aspect_ratios = []
    
    for img_path in image_paths[:1000]:  # Sample first 1000 images
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    stats = {
        'count': len(image_paths),
        'sample_size': len(widths),
        'width': {
            'min': min(widths),
            'max': max(widths),
            'mean': np.mean(widths),
            'std': np.std(widths)
        },
        'height': {
            'min': min(heights),
            'max': max(heights),
            'mean': np.mean(heights),
            'std': np.std(heights)
        },
        'aspect_ratio': {
            'min': min(aspect_ratios),
            'max': max(aspect_ratios),
            'mean': np.mean(aspect_ratios),
            'std': np.std(aspect_ratios)
        }
    }
    
    return stats

def analyze_class_distribution(data_dir):
    class_counts = {}
    
    # 从训练集目录中分析类别分布
    train_images_dir = os.path.join(data_dir, 'images')
    if os.path.exists(train_images_dir):
        for filename in os.listdir(train_images_dir):
            if os.path.splitext(filename)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
                try:
                    class_id = int(filename.split('_')[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                except (ValueError, IndexError):
                    print(f"Warning: Could not extract class from filename: {filename}")
    
    # 如果有单独的验证集目录，也分析其类别分布
    # 这里假设验证集目录结构与训练集相同
    # 注意：如果验证集目录与训练集目录相同，则不需要重复计算
    
    sorted_counts = sorted(class_counts.items())
    
    return {
        'class_counts': dict(sorted_counts),
        'total_samples': sum(class_counts.values()),
        'num_classes': len(class_counts),
        'min_samples': min(class_counts.values()) if class_counts else 0,
        'max_samples': max(class_counts.values()) if class_counts else 0
    }

def plot_class_distribution(class_counts, save_path=None):
    labels = [f"Class_{k}" for k in sorted(class_counts.keys())]
    counts = [class_counts[k] for k in sorted(class_counts.keys())]
    
    plt.figure(figsize=(15, 6))
    plt.bar(labels, counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_model_summary(model, input_size=(3, 224, 224)):
    from torchsummary import summary
    
    device = get_device()
    model = model.to(device)
    
    print(model)
    summary(model, input_size)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

def create_results_directory(base_dir='./results'):
    os.makedirs(base_dir, exist_ok=True)
    
    subdirs = ['checkpoints', 'plots', 'logs', 'predictions']
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    return base_dir

def log_experiment(config, results, log_file='experiment_log.json'):
    log_entry = {
        'timestamp': str(torch.datetime.now()),
        'config': config,
        'results': results
    }
    
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    
    logs.append(log_entry)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"Experiment logged to {log_file}")