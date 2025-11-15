import os
import torch
import torch.nn.functional as F
from PIL import Image
import json
import argparse
from pathlib import Path
import torchvision.transforms as transforms

from model import create_model
from utils import load_checkpoint, analyze_class_distribution
from config import Config

class DiseasePredictor:
    def __init__(self, model_path, model_type='yolo11x-cls', num_classes=None, data_dir=None, device=None):
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 如果没有指定类别数量，尝试从数据目录分析
        if num_classes is None and data_dir is not None:
            class_dist = analyze_class_distribution(data_dir)
            num_classes = class_dist['num_classes']
            print(f"Detected {num_classes} classes from data directory")
        
        # 如果仍然没有类别数量，使用默认值
        if num_classes is None:
            num_classes = 61
            print(f"Using default number of classes: {num_classes}")
            
        self.num_classes = num_classes
        self.model = create_model(model_type=model_type, num_classes=num_classes, pretrained=False)
        self.model = self.model.to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = load_checkpoint(model_path)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found")
        
        self.model.eval()
        self.transform = self._build_transform()
        
        self.class_names = self._load_class_names()

    def _build_transform(self):
        return transforms.Compose([
            transforms.Resize(Config.IMG_SIZE + 32),
            transforms.CenterCrop(Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_class_names(self):
        class_names = [f"Class_{i}" for i in range(self.num_classes)]
        
        class_mapping_file = 'class_mapping.json'
        if os.path.exists(class_mapping_file):
            with open(class_mapping_file, 'r') as f:
                mapping = json.load(f)
                if isinstance(mapping, dict):
                    class_names = [mapping.get(str(i), f"Class_{i}") for i in range(self.num_classes)]
                elif isinstance(mapping, list):
                    class_names = mapping
        
        return class_names
    
    def predict_single_image(self, image_path, top_k=5):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # 确保outputs是一个张量，如果是元组则取第一个元素
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probabilities = F.softmax(outputs, dim=1)
            
            top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))
            
            results = []
            valid_k = top_probs.size(1)
            for i in range(valid_k):
                class_idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                class_name = self.class_names[class_idx]
                
                results.append({
                    'class_id': class_idx,
                    'class_name': class_name,
                    'probability': prob
                })
        
        return results
    
    def predict_batch(self, image_paths, top_k=5):
        results = []
        for image_path in image_paths:
            result = self.predict_single_image(image_path, top_k)
            results.append({
                'image_path': image_path,
                'predictions': result
            })
        return results
    
    def predict_directory(self, directory_path, output_file=None, top_k=5):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(directory_path).glob(f'*{ext}'))
            image_paths.extend(Path(directory_path).glob(f'*{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        
        print(f"Found {len(image_paths)} images in {directory_path}")
        
        results = self.predict_batch(image_paths, top_k)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results
    
    def predict_with_confidence(self, image_path, confidence_threshold=0.5):
        results = self.predict_single_image(image_path, top_k=1)
        
        if results and results[0]['probability'] >= confidence_threshold:
            return results[0]
        else:
            return {
                'class_id': -1,
                'class_name': 'Unknown',
                'probability': 0.0,
                'message': f'Confidence below threshold {confidence_threshold}'
            }
    
    def visualize_prediction(self, image_path, save_path=None):
        import matplotlib.pyplot as plt
        
        results = self.predict_single_image(image_path, top_k=3)
        
        image = Image.open(image_path).convert('RGB')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(image)
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        if results:
            classes = [r['class_name'] for r in results]
            probs = [r['probability'] for r in results]
            
            bars = ax2.barh(classes, probs)
            ax2.set_xlim(0, 1)
            ax2.set_xlabel('Probability')
            ax2.set_title('Top Predictions')
            
            for i, (bar, prob) in enumerate(zip(bars, probs)):
                ax2.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Crop Disease Prediction')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth',
                        help='Path to the trained model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output file for batch predictions')
    parser.add_argument('--model_type', type=str, default='yolo11x-cls',
                        help='Model architecture')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to return')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize prediction results')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    
    args = parser.parse_args()
    
    # 尝试从模型路径推断数据目录
    data_dir = None
    if os.path.exists(args.model_path):
        # 假设模型在checkpoints目录下，数据在上一级目录
        model_dir = os.path.dirname(args.model_path)
        if model_dir.endswith('checkpoints'):
            data_dir = os.path.dirname(model_dir)
    
    predictor = DiseasePredictor(args.model_path, args.model_type, data_dir=data_dir)
    
    if os.path.isfile(args.input):
        if args.visualize:
            output_path = args.output if args.output else 'prediction_visualization.png'
            predictor.visualize_prediction(args.input, output_path)
        else:
            result = predictor.predict_with_confidence(args.input, args.confidence)
            print(f"Prediction for {args.input}:")
            print(f"Class: {result['class_name']} (ID: {result['class_id']})")
            print(f"Probability: {result['probability']:.4f}")
            if 'message' in result:
                print(f"Note: {result['message']}")
    
    elif os.path.isdir(args.input):
        results = predictor.predict_directory(args.input, args.output, args.top_k)
        
        for result in results[:5]:  # Print first 5 results
            image_path = result['image_path']
            predictions = result['predictions']
            top_pred = predictions[0] if predictions else None
            
            if top_pred:
                print(f"{os.path.basename(image_path)}: {top_pred['class_name']} ({top_pred['probability']:.4f})")
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main()
