"""
Utility functions for car detection
"""

import os
import json
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_yaml_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_yaml_config(config: Dict, output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Numpy array of image or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save image to file.
    
    Args:
        image: Numpy array of image
        output_path: Path to save image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert RGB back to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
        return True
    except Exception as e:
        print(f"Error saving image {output_path}: {e}")
        return False


def draw_boxes(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image
        detections: List of detection dictionaries with 'box' and optional 'conf'
        
    Returns:
        Image with drawn boxes
    """
    result = image.copy()
    
    for detection in detections:
        if 'box' not in detection:
            continue
            
        x1, y1, x2, y2 = detection['box']
        conf = detection.get('conf', 1.0)
        label = detection.get('label', 'car')
        
        # Draw box
        cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), 
                     (0, 255, 0), 2)
        
        # Draw label
        text = f"{label}: {conf:.2f}"
        cv2.putText(result, text, (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result


def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get list of image files from directory.
    
    Args:
        directory: Path to directory
        extensions: List of file extensions to search for
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    image_files = []
    for ext in extensions:
        image_files.extend(Path(directory).glob(f'*{ext}'))
        image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
    
    return sorted([str(f) for f in image_files])


def create_directory(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def validate_dataset_structure(data_dir: str) -> bool:
    """
    Validate dataset directory structure.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        True if structure is valid, False otherwise
    """
    required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    
    for req_dir in required_dirs:
        path = os.path.join(data_dir, req_dir)
        if not os.path.isdir(path):
            print(f"Missing directory: {path}")
            return False
    
    print("Dataset structure is valid")
    return True
