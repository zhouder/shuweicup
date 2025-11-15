import os
import torch
import argparse
import json
import numpy as np
from sklearn.metrics import classification_report

from config import Config
from data_loader_v2 import get_data_loaders, analyze_class_distribution
from models.yolo11_classifier import YOLO11Classifier
from models.resnet_baseline import ResNet18Baseline
from visualizer import plot_model_comparison, plot_per_class_metrics


def evaluate_model(model, val_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1_score': report['macro avg']['f1-score']
    }
    
    return metrics, report, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description='Compare YOLO11x-cls and ResNet18 models')
    parser.add_argument('--yolo_checkpoint', type=str, default='./checkpoints/best_model_f1.pth',
                        help='Path to YOLO11x-cls checkpoint')
    parser.add_argument('--resnet_checkpoint', type=str, default='./checkpoints/resnet_best_model_f1.pth',
                        help='Path to ResNet18 checkpoint')
    parser.add_argument('--train_dir', type=str, default=Config.DATA_DIR,
                        help='Training data directory')
    parser.add_argument('--val_dir', type=str, default=Config.VAL_DIR,
                        help='Validation data directory')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                        help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Analyzing dataset...")
    class_dist = analyze_class_distribution(args.train_dir)
    num_classes = class_dist['num_classes']
    print(f"Found {num_classes} classes")
    
    print("Creating data loaders...")
    config = Config()
    config.DATA_DIR = args.train_dir
    config.VAL_DIR = args.val_dir
    config.NUM_CLASSES = num_classes
    
    _, val_loader, _, _ = get_data_loaders(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = {}
    
    print("\n=== Evaluating YOLO11x-cls ===")
    yolo_model = YOLO11Classifier(num_classes, pretrained=False)
    
    if os.path.exists(args.yolo_checkpoint):
        yolo_model.load_state_dict(torch.load(args.yolo_checkpoint, map_location=device))
        print(f"Loaded YOLO11x-cls checkpoint from {args.yolo_checkpoint}")
    else:
        print(f"Warning: YOLO11x-cls checkpoint not found at {args.yolo_checkpoint}")
        print("Using randomly initialized model")
    
    yolo_model.to(device)
    yolo_metrics, yolo_report, yolo_labels, yolo_preds = evaluate_model(yolo_model, val_loader, device)
    results['YOLO11x-cls'] = yolo_metrics
    
    print(f"YOLO11x-cls Results:")
    print(f"  Accuracy: {yolo_metrics['accuracy']:.4f}")
    print(f"  Precision: {yolo_metrics['precision']:.4f}")
    print(f"  Recall: {yolo_metrics['recall']:.4f}")
    print(f"  F1-Score: {yolo_metrics['f1_score']:.4f}")
    
    print("\n=== Evaluating ResNet18 ===")
    resnet_model = ResNet18Baseline(num_classes, pretrained=False)
    
    if os.path.exists(args.resnet_checkpoint):
        resnet_model.load_state_dict(torch.load(args.resnet_checkpoint, map_location=device))
        print(f"Loaded ResNet18 checkpoint from {args.resnet_checkpoint}")
    else:
        print(f"Warning: ResNet18 checkpoint not found at {args.resnet_checkpoint}")
        print("Using randomly initialized model")
    
    resnet_model.to(device)
    resnet_metrics, resnet_report, resnet_labels, resnet_preds = evaluate_model(resnet_model, val_loader, device)
    results['ResNet18'] = resnet_metrics
    
    print(f"ResNet18 Results:")
    print(f"  Accuracy: {resnet_metrics['accuracy']:.4f}")
    print(f"  Precision: {resnet_metrics['precision']:.4f}")
    print(f"  Recall: {resnet_metrics['recall']:.4f}")
    print(f"  F1-Score: {resnet_metrics['f1_score']:.4f}")
    
    print("\n=== Comparison Summary ===")
    print(f"{'Metric':<12} {'YOLO11x-cls':<12} {'ResNet18':<12} {'Winner':<12}")
    print("-" * 48)
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        yolo_val = yolo_metrics[metric]
        resnet_val = resnet_metrics[metric]
        winner = 'YOLO11x-cls' if yolo_val > resnet_val else 'ResNet18'
        print(f"{metric:<12} {yolo_val:<12.4f} {resnet_val:<12.4f} {winner:<12}")
    
    improvement = ((yolo_metrics['f1_score'] - resnet_metrics['f1_score']) / resnet_metrics['f1_score']) * 100
    print(f"\nF1-Score Improvement: {improvement:.2f}%")
    
    print("\n=== Saving Results ===")
    
    with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w') as f:
        json.dump({
            'yolo11x_cls': {
                'metrics': yolo_metrics,
                'report': yolo_report
            },
            'resnet18': {
                'metrics': resnet_metrics,
                'report': resnet_report
            }
        }, f, indent=2)
    
    plot_model_comparison(results, args.output_dir)
    
    class_names = [f'Class {i}' for i in range(num_classes)]
    plot_per_class_metrics(yolo_report, args.output_dir, class_names)
    
    print(f"Results saved to {args.output_dir}")
    
    return results


if __name__ == "__main__":
    main()