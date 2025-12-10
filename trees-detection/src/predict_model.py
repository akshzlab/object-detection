"""
Inference script for YOLOv8 tree detection model.

This script runs tree detection inference on images or video files.
"""
import os
import sys
import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import setup_logging


class TreeDetector:
    """
    Tree detection inference engine.
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, logger=None):
        """
        Initialize the tree detector.
        
        Args:
            model_path: Path to trained YOLO model weights
            conf_threshold: Confidence threshold for detections (0-1)
            logger: Logger instance
        """
        self.logger = logger or setup_logging()
        self.conf_threshold = conf_threshold
        
        # Load model
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
    
    def detect(self, source: str, conf: float = None) -> list:
        """
        Run tree detection on an image or video.
        
        Args:
            source: Path to image or video file
            conf: Override confidence threshold
            
        Returns:
            List of detection results
        """
        conf = conf or self.conf_threshold
        
        if not os.path.exists(source):
            self.logger.error(f"Source file not found: {source}")
            return []
        
        self.logger.info(f"Running inference on {source}")
        
        try:
            results = self.model.predict(
                source=source,
                conf=conf,
                verbose=False
            )
            self.logger.info(f"Inference complete. Detected {len(results)} frame(s)")
            return results
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return []
    
    def detect_and_save(self, source: str, output_dir: str = 'output', 
                       conf: float = None, save_crops: bool = False) -> list:
        """
        Run tree detection and save results.
        
        Args:
            source: Path to image or video file
            output_dir: Directory to save results
            conf: Override confidence threshold
            save_crops: Whether to save cropped tree images
            
        Returns:
            List of detection results
        """
        conf = conf or self.conf_threshold
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Running inference with output to {output_dir}")
        
        try:
            results = self.model.predict(
                source=source,
                conf=conf,
                save=True,
                save_crop=save_crops,
                project=output_dir,
                name='predictions',
                exist_ok=True,
                verbose=False
            )
            
            self.logger.info(f"Results saved to {output_dir}/predictions")
            
            # Print detection summary
            for i, result in enumerate(results):
                tree_count = len(result.boxes)
                self.logger.info(f"Frame {i}: Detected {tree_count} trees")
            
            return results
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return []
    
    def count_trees(self, source: str, conf: float = None) -> dict:
        """
        Count trees in an image or video.
        
        Args:
            source: Path to image or video file
            conf: Override confidence threshold
            
        Returns:
            Dictionary with tree counts
        """
        results = self.detect(source, conf)
        
        if not results:
            return {'total_trees': 0, 'details': []}
        
        details = []
        total_trees = 0
        
        for i, result in enumerate(results):
            tree_count = len(result.boxes)
            total_trees += tree_count
            details.append({
                'frame': i,
                'tree_count': tree_count,
                'confidences': result.boxes.conf.cpu().numpy().tolist()
            })
        
        return {
            'total_trees': total_trees,
            'frame_count': len(results),
            'details': details
        }


def main():
    """Main entry point for inference script."""
    parser = argparse.ArgumentParser(
        description='Run tree detection inference on images/videos'
    )
    parser.add_argument(
        'source',
        type=str,
        help='Path to image or video file'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained YOLO model weights (.pt file)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (0-1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--save-crops',
        action='store_true',
        help='Save cropped tree images'
    )
    parser.add_argument(
        '--count-only',
        action='store_true',
        help='Only count trees, don\'t save images'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Initialize detector
    try:
        detector = TreeDetector(args.model, conf_threshold=args.conf, logger=logger)
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return 1
    
    # Run inference
    if args.count_only:
        logger.info("Running in count-only mode")
        stats = detector.count_trees(args.source, conf=args.conf)
        logger.info(f"Total trees detected: {stats['total_trees']}")
        for detail in stats['details']:
            logger.info(f"  Frame {detail['frame']}: {detail['tree_count']} trees")
    else:
        results = detector.detect_and_save(
            args.source,
            output_dir=args.output,
            conf=args.conf,
            save_crops=args.save_crops
        )
        
        if results:
            logger.info(f"Inference results saved to {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
