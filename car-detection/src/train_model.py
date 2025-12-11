"""
Train YOLO model for car detection
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Optional
import torch
from ultralytics import YOLO
import yaml


def setup_training(config_path: str, project_dir: str = "runs") -> Dict:
    """
    Setup training configuration.
    
    Args:
        config_path: Path to data configuration YAML
        project_dir: Directory to save training results
        
    Returns:
        Training configuration dictionary
    """
    with open(config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    return {
        'data_config': data_config,
        'project_dir': project_dir
    }


def train_yolo_model(
    config_path: str,
    model_name: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    device: int = 0,
    project: str = "runs/detect",
    name: str = "car_detection",
    patience: int = 20,
    resume: bool = False
) -> YOLO:
    """
    Train YOLO model for car detection.
    
    Args:
        config_path: Path to data configuration YAML file
        model_name: YOLO model to use (e.g., 'yolov8n.pt', 'yolov8s.pt')
        epochs: Number of training epochs
        imgsz: Image size for training
        batch_size: Batch size for training
        device: GPU device ID (0 for GPU, 'cpu' for CPU)
        project: Project directory for results
        name: Name of the experiment
        patience: Early stopping patience
        resume: Resume training from checkpoint
        
    Returns:
        Trained YOLO model
    """
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Load YOLO model
    model = YOLO(model_name)
    
    # Train model
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save=True,
        save_period=10,
        verbose=True,
        resume=resume,
        # Data augmentation
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        # Optimization
        optimizer='SGD',
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
    )
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {project}/{name}")
    
    return model


def validate_model(model: YOLO, data_config: str) -> Dict:
    """
    Validate model on validation set.
    
    Args:
        model: Trained YOLO model
        data_config: Path to data configuration YAML
        
    Returns:
        Validation metrics dictionary
    """
    print("\nValidating model...")
    metrics = model.val(data=data_config)
    
    print(f"Validation Results:")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    
    return metrics


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train YOLO model for car detection')
    
    parser.add_argument('--config', type=str, default='config/car_data.yaml',
                       help='Path to data configuration YAML')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model to use (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID (0 for GPU, -1 for CPU)')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Project directory for results')
    parser.add_argument('--name', type=str, default='car_detection',
                       help='Name of the experiment')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Determine device
    device = 'cpu' if args.device == -1 else args.device
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return
    
    # Train model
    model = train_yolo_model(
        config_path=args.config,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume
    )
    
    # Validate model
    validate_model(model, args.config)


if __name__ == '__main__':
    main()
