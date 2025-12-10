"""
Model evaluation script for YOLOv8 tree detection.

This script evaluates model performance and generates metrics.
"""
import os
import sys
import argparse
import json
from pathlib import Path
from ultralytics import YOLO

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import setup_logging, load_yaml_config


class ModelEvaluator:
    """
    Evaluate YOLOv8 model performance.
    """
    
    def __init__(self, model_path: str, logger=None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model weights
            logger: Logger instance
        """
        self.logger = logger or setup_logging()
        
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.logger.info(f"Loading model from {model_path}")
        try:
            self.model = YOLO(model_path)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate(self, data_config: str, imgsz: int = 640) -> dict:
        """
        Evaluate model on validation set.
        
        Args:
            data_config: Path to YOLO data configuration file
            imgsz: Image size for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        if not os.path.exists(data_config):
            self.logger.error(f"Data config not found: {data_config}")
            return None
        
        self.logger.info("=" * 80)
        self.logger.info("Starting Model Evaluation")
        self.logger.info("=" * 80)
        
        try:
            results = self.model.val(
                data=data_config,
                imgsz=imgsz,
                verbose=True
            )
            
            self.logger.info("=" * 80)
            self.logger.info("Evaluation Complete!")
            self.logger.info("=" * 80)
            
            return {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.p[0]),
                'recall': float(results.box.r[0]),
                'fitness': float(results.box.fitness)
            }
        
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            self.logger.exception("Full traceback:")
            return None
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model info
        """
        try:
            info = {
                'parameters': self.model.model.parameters(),
                'layers': len(self.model.model),
            }
            return info
        except:
            return {}


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description='Evaluate YOLOv8 tree detection model'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model weights (.pt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YOLO data configuration file'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for evaluation'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='runs/evaluate',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Initialize evaluator
    try:
        evaluator = ModelEvaluator(args.model, logger=logger)
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return 1
    
    # Evaluate
    results = evaluator.evaluate(args.config, imgsz=args.imgsz)
    
    if results is None:
        return 1
    
    # Print results
    logger.info("\nEvaluation Metrics:")
    logger.info(f"  mAP50:     {results['mAP50']:.4f}")
    logger.info(f"  mAP50-95:  {results['mAP50-95']:.4f}")
    logger.info(f"  Precision: {results['precision']:.4f}")
    logger.info(f"  Recall:    {results['recall']:.4f}")
    logger.info(f"  Fitness:   {results['fitness']:.4f}")
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    results_file = os.path.join(args.output, 'eval_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
