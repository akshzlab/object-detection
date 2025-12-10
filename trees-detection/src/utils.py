"""
Utility functions for tree detection system.
"""
import os
import logging
from pathlib import Path
import yaml


def setup_logging(log_file: str = None) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_file: Path to log file (optional)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_yaml_config(config_path: str) -> dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_data_structure(data_path: str, logger: logging.Logger = None) -> bool:
    """
    Validate that data directory follows YOLO format.
    Expected structure:
    - data/
      - images/
        - *.jpg, *.png
      - labels/
        - *.txt (YOLO format)
    
    Args:
        data_path: Path to data directory
        logger: Logger instance
        
    Returns:
        True if valid, False otherwise
    """
    if logger is None:
        logger = setup_logging()
    
    data_dir = Path(data_path)
    
    # Check if data directory exists
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_path}")
        return False
    
    # Check required subdirectories
    images_dir = data_dir / 'images'
    labels_dir = data_dir / 'labels'
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return False
    
    if not labels_dir.exists():
        logger.warning(f"Labels directory not found: {labels_dir}")
        logger.warning("Labels may not be available for all datasets")
    
    # Count images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = [f for f in images_dir.iterdir() 
              if f.suffix.lower() in image_extensions]
    
    if not images:
        logger.warning(f"No images found in {images_dir}")
        return False
    
    logger.info(f"Found {len(images)} images in {images_dir}")
    
    return True


def create_directory_structure(base_path: str, logger: logging.Logger = None) -> None:
    """
    Create standard directory structure for the project.
    
    Args:
        base_path: Base path for the project
        logger: Logger instance
    """
    if logger is None:
        logger = setup_logging()
    
    dirs = [
        'data/raw',
        'data/processed',
        'data/train/images',
        'data/train/labels',
        'data/val/images',
        'data/val/labels',
        'models',
        'runs/detect',
        'logs'
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        logger.debug(f"Directory ready: {full_path}")
    
    logger.info(f"Directory structure created at {base_path}")


def get_absolute_path(relative_path: str, base_dir: str = None) -> str:
    """
    Convert relative path to absolute path.
    
    Args:
        relative_path: Relative path
        base_dir: Base directory (defaults to script directory)
        
    Returns:
        Absolute path
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.abspath(os.path.join(base_dir, relative_path))
