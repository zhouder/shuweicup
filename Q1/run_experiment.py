import os
import argparse
import subprocess
import sys


def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run Crop Disease Recognition Experiments')
    parser.add_argument('--mode', type=str, choices=['train_yolo', 'train_resnet', 'compare', 'all'], 
                        default='all', help='Experiment mode')
    parser.add_argument('--train_dir', type=str, default='../AgriculturalDisease_trainingset',
                        help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='../AgriculturalDisease_validationset',
                        help='Validation data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use exponential moving average')
    
    args = parser.parse_args()
    
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./comparison_results', exist_ok=True)
    
    base_args = [
        '--train_dir', args.train_dir,
        '--val_dir', args.val_dir,
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr)
    ]
    
    if args.use_amp:
        base_args.append('--use_amp')
    
    if args.use_ema:
        base_args.append('--use_ema')
    
    success = True
    
    if args.mode in ['train_yolo', 'all']:
        print("\n=== Training YOLO11x-cls ===")
        yolo_args = ['python', 'train_v2.py', '--model_type', 'yolo11x-cls'] + base_args
        success &= run_command(yolo_args)
        
        if success:
            print("YOLO11x-cls training completed successfully")
        else:
            print("YOLO11x-cls training failed")
            sys.exit(1)
    
    if args.mode in ['train_resnet', 'all']:
        print("\n=== Training ResNet18 ===")
        resnet_args = ['python', 'train_v2.py', '--model_type', 'resnet18'] + base_args
        success &= run_command(resnet_args)
        
        if success:
            print("ResNet18 training completed successfully")
        else:
            print("ResNet18 training failed")
            sys.exit(1)
    
    if args.mode in ['compare', 'all']:
        print("\n=== Comparing Models ===")
        compare_args = [
            'python', 'compare_models.py',
            '--train_dir', args.train_dir,
            '--val_dir', args.val_dir,
            '--output_dir', './comparison_results'
        ]
        success &= run_command(compare_args)
        
        if success:
            print("Model comparison completed successfully")
        else:
            print("Model comparison failed")
            sys.exit(1)
    
    if success:
        print("\n=== All experiments completed successfully! ===")
        print("Results saved to:")
        print("  - Model checkpoints: ./checkpoints/")
        print("  - Training figures: ./figures/")
        print("  - Comparison results: ./comparison_results/")
    else:
        print("\n=== Some experiments failed ===")
        sys.exit(1)


if __name__ == "__main__":
    main()