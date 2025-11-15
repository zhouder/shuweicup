import os
import argparse
from utils import create_class_mapping, analyze_class_distribution, get_image_statistics

def setup_environment():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
def analyze_data(train_dir, val_dir):
    print("Analyzing dataset...")
    
    class_dist = analyze_class_distribution(train_dir)
    print(f"Total samples: {class_dist['total_samples']}")
    print(f"Number of classes: {class_dist['num_classes']}")
    print(f"Min samples per class: {class_dist['min_samples']}")
    print(f"Max samples per class: {class_dist['max_samples']}")
    
    img_stats = get_image_statistics(train_dir)
    if img_stats:
        print(f"\nImage statistics (based on {img_stats['sample_size']} samples):")
        print(f"Width: {img_stats['width']['min']}-{img_stats['width']['max']} (mean: {img_stats['width']['mean']:.1f})")
        print(f"Height: {img_stats['height']['min']}-{img_stats['height']['max']} (mean: {img_stats['height']['mean']:.1f})")
    
    # 创建类别映射，需要分别处理训练集和验证集
    train_mapping = create_class_mapping(train_dir, 'train_class_mapping.json')
    val_mapping = create_class_mapping(val_dir, 'val_class_mapping.json')
    print(f"\nCreated class mapping with {len(train_mapping)} training classes and {len(val_mapping)} validation classes")
    
    return class_dist, img_stats, train_mapping

def main():
    parser = argparse.ArgumentParser(description='Crop Disease Recognition System')
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'analyze'], 
                        default='analyze', help='Mode to run')
    parser.add_argument('--train_dir', type=str, default='../AgriculturalDisease_trainingset',
                        help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='../AgriculturalDisease_validationset',
                        help='Validation data directory')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--input', type=str, help='Input image or directory for inference')
    parser.add_argument('--output', type=str, help='Output file for inference results')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    setup_environment()
    
    if args.mode == 'analyze':
        analyze_data(args.train_dir, args.val_dir)
    
    elif args.mode == 'train':
        print("Starting training...")
        from train import main as train_main
        
        import sys
        sys.argv = [
            'train.py',
            '--train_dir', args.train_dir,
            '--val_dir', args.val_dir,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr)
        ]
        
        train_main()
    
    elif args.mode == 'inference':
        if not args.input:
            print("Error: --input is required for inference mode")
            return
        
        print("Starting inference...")
        from inference import main as infer_main
        
        import sys
        sys.argv = [
            'inference.py',
            '--model_path', args.model_path,
            '--input', args.input
        ]
        
        if args.output:
            sys.argv.extend(['--output', args.output])
        
        infer_main()

if __name__ == "__main__":
    main()