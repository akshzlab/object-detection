"""
Training script for YOLOv8 tree detection model.

This script trains a YOLOv8 model on tree detection data using transfer learning.
"""
import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import setup_logging, load_yaml_config, validate_data_structure


def train_tree_detector(config_path: str, model_size: str = 'n', epochs: int = 100, 
                        batch_size: int = 16, patience: int = 20, device: str = 0):
    """
    Train a YOLOv8 model for tree detection using transfer learning.
    
    Args:
        config_path: Path to YOLO data configuration file
        model_size: Model size ('n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=xlarge)
        epochs: Number of training epochs
        batch_size: Batch size for training
        patience: Early stopping patience (epochs without improvement)
        device: GPU device ID (0 for first GPU, 'cpu' for CPU)
    """
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Tree Detection - YOLOv8 Training Pipeline")
    logger.info("=" * 80)
    
    # Validate config file exists
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return None
    
    # Load and validate config
    try:
        config = load_yaml_config(config_path)
        logger.info(f"Loaded config from {config_path}")
        logger.info(f"Config: {config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None
    
    # Validate data structure
    data_path = config.get('path')
    if data_path and not os.path.isabs(data_path):
        # Make path absolute relative to config file location
        config_dir = os.path.dirname(os.path.abspath(config_path))
        data_path = os.path.join(config_dir, data_path)
    
    logger.info(f"Data path: {data_path}")
    if not validate_data_structure(data_path, logger):
        logger.warning("Data structure validation issues detected. Continuing anyway...")
    
    # Load pre-trained model
    logger.info(f"Loading YOLOv8{model_size} pre-trained model...")
    try:
        model = YOLO(f'yolov8{model_size}.pt')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
    
    # Training parameters
    logger.info("Training parameters:")
    logger.info(f"  - Epochs: {epochs}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Patience: {patience}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Model size: {model_size}")
    
    # Train model
    logger.info("Starting training...")
    try:
        results = model.train(
            data=config_path,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            patience=patience,
            device=device,
            name='tree_model_v1',
            project='runs/detect',
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            exist_ok=True,
            verbose=True
        )
        
        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info(f"Best model saved at: {results.save_dir}")
        logger.info("=" * 80)
        
        return results
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Full traceback:")
        return None


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 model for tree detection'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/tree_data.yaml',
        help='Path to YOLO data configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='GPU device ID or "cpu"'
    )
    
    args = parser.parse_args()
    
    # Convert config path to absolute if needed
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)
    
    # Run training
    results = train_tree_detector(
        config_path=config_path,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        patience=args.patience,
        device=args.device
    )
    
    return 0 if results is not None else 1


if __name__ == '__main__':
    sys.exit(main())
