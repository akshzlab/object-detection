"""
Predict car detections using trained YOLO model
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from utils import load_image, save_image, draw_boxes, get_image_files


class CarDetector:
    """Car detection using YOLO"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        Initialize car detector.
        
        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect cars in image.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Tuple of (annotated image, list of detections)
        """
        # Convert to BGR for YOLO (it expects BGR)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = self.model(image_bgr, conf=self.conf_threshold, verbose=False)
        
        detections = []
        annotated = image.copy()
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                
                detection = {
                    'box': [x1, y1, x2, y2],
                    'conf': conf,
                    'class_id': cls_id,
                    'label': 'car'
                }
                detections.append(detection)
            
            # Draw boxes on image
            annotated = draw_boxes(image, detections)
        
        return annotated, detections
    
    def detect_batch(self, image_paths: List[str]) -> Dict[str, Tuple[np.ndarray, List[Dict]]]:
        """
        Detect cars in batch of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping image paths to (annotated_image, detections)
        """
        results = {}
        
        for image_path in image_paths:
            print(f"Processing: {image_path}")
            
            image = load_image(image_path)
            if image is None:
                continue
            
            annotated, detections = self.detect(image)
            results[image_path] = (annotated, detections)
        
        return results


def predict_image(
    model_path: str,
    image_path: str,
    output_path: Optional[str] = None,
    conf_threshold: float = 0.5
) -> List[Dict]:
    """
    Predict cars in a single image.
    
    Args:
        model_path: Path to trained YOLO model
        image_path: Path to input image
        output_path: Path to save annotated image (optional)
        conf_threshold: Confidence threshold for detections
        
    Returns:
        List of detections
    """
    # Load image
    image = load_image(image_path)
    if image is None:
        return []
    
    # Initialize detector
    detector = CarDetector(model_path, conf_threshold)
    
    # Detect
    annotated, detections = detector.detect(image)
    
    # Save result
    if output_path:
        save_image(annotated, output_path)
        print(f"Result saved to: {output_path}")
    
    # Print detections
    print(f"\nDetected {len(detections)} cars:")
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det['box']
        conf = det['conf']
        print(f"  {i}. Car: confidence={conf:.3f}, box=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    
    return detections


def predict_directory(
    model_path: str,
    image_dir: str,
    output_dir: str = "predictions",
    conf_threshold: float = 0.5
) -> Dict:
    """
    Predict cars in all images in a directory.
    
    Args:
        model_path: Path to trained YOLO model
        image_dir: Path to directory containing images
        output_dir: Path to save annotated images
        conf_threshold: Confidence threshold for detections
        
    Returns:
        Dictionary of results
    """
    # Get image files
    image_files = get_image_files(image_dir)
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return {}
    
    print(f"Found {len(image_files)} images")
    
    # Initialize detector
    detector = CarDetector(model_path, conf_threshold)
    
    # Process all images
    all_detections = {}
    os.makedirs(output_dir, exist_ok=True)
    
    for image_path in image_files:
        image = load_image(image_path)
        if image is None:
            continue
        
        annotated, detections = detector.detect(image)
        all_detections[image_path] = detections
        
        # Save annotated image
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        save_image(annotated, output_path)
        
        print(f"  {filename}: {len(detections)} cars detected")
    
    print(f"\nPredictions saved to: {output_dir}")
    return all_detections


def main():
    """Main prediction script"""
    parser = argparse.ArgumentParser(description='Predict car detections using YOLO')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLO model')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image or directory of images')
    parser.add_argument('--output', type=str, default='predictions',
                       help='Path to save results')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return
    
    # Check if source exists
    if not os.path.exists(args.source):
        print(f"Error: Source not found: {args.source}")
        return
    
    # Process based on source type
    if os.path.isfile(args.source):
        # Single image
        output_path = os.path.join(args.output, 
                                  os.path.basename(args.source))
        predict_image(args.model, args.source, output_path, args.conf)
    else:
        # Directory of images
        predict_directory(args.model, args.source, args.output, args.conf)


if __name__ == '__main__':
    main()
