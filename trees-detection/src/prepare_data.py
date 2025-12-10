"""
Data preparation script for YOLOv8 tree detection.

This script helps organize, validate, and prepare data for training.
"""
import os
import sys
import argparse
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import setup_logging, create_directory_structure


def organize_yolo_format_data(images_dir: str, labels_dir: str, output_dir: str, 
                              train_ratio: float = 0.8, val_ratio: float = 0.1, 
                              logger=None) -> bool:
    """
    Organize images and labels into YOLO train/val/test split.
    
    Expected input structure:
    - images_dir/
      - *.jpg, *.png
    - labels_dir/
      - *.txt (YOLO format, matching image names)
    
    Args:
        images_dir: Path to directory containing all images
        labels_dir: Path to directory containing all labels
        output_dir: Path to output directory for organized data
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    if logger is None:
        logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("Organizing data into YOLO format")
    logger.info("=" * 80)
    
    # Validate input directories
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    if not images_path.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return False
    
    if not labels_path.exists():
        logger.warning(f"Labels directory not found: {labels_dir}")
        logger.warning("Proceeding with images only (no labels)")
    
    # Get list of images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in images_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.error(f"No images found in {images_dir}")
        return False
    
    logger.info(f"Found {len(image_files)} images")
    
    # Create output structure
    output_path = Path(output_dir)
    create_directory_structure(str(output_path), logger)
    
    # Split data
    train_ratio = train_ratio / (train_ratio + val_ratio)
    train_files, val_files = train_test_split(
        image_files,
        train_size=train_ratio,
        random_state=42
    )
    
    logger.info(f"Train/Val split: {len(train_files)}/{len(val_files)}")
    
    # Copy files
    def copy_split(file_list, split_name):
        """Copy files to split directory."""
        for image_file in file_list:
            # Copy image
            dest_img = output_path / split_name / 'images' / image_file.name
            shutil.copy2(image_file, dest_img)
            
            # Copy label if exists
            label_file = labels_path / (image_file.stem + '.txt')
            if label_file.exists():
                dest_label = output_path / split_name / 'labels' / label_file.name
                shutil.copy2(label_file, dest_label)
        
        logger.info(f"Copied {len(file_list)} files to {split_name}")
    
    copy_split(train_files, 'train')
    copy_split(val_files, 'val')
    
    logger.info("=" * 80)
    logger.info("Data organization complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    
    return True


def validate_yolo_format(data_dir: str, logger=None) -> dict:
    """
    Validate YOLO format data structure and count files.
    
    Args:
        data_dir: Path to data directory
        logger: Logger instance
        
    Returns:
        Dictionary with validation results
    """
    if logger is None:
        logger = setup_logging()
    
    data_path = Path(data_dir)
    results = {
        'valid': True,
        'train': {'images': 0, 'labels': 0},
        'val': {'images': 0, 'labels': 0},
        'issues': []
    }
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for split in ['train', 'val']:
        split_path = data_path / split
        
        if not split_path.exists():
            results['issues'].append(f"{split} directory not found")
            results['valid'] = False
            continue
        
        # Count images
        images_dir = split_path / 'images'
        if images_dir.exists():
            images = [f for f in images_dir.iterdir() 
                     if f.suffix.lower() in image_extensions]
            results[split]['images'] = len(images)
        else:
            results['issues'].append(f"{split}/images directory not found")
            results['valid'] = False
        
        # Count labels
        labels_dir = split_path / 'labels'
        if labels_dir.exists():
            labels = [f for f in labels_dir.iterdir() if f.suffix == '.txt']
            results[split]['labels'] = len(labels)
        else:
            results['issues'].append(f"{split}/labels directory not found")
    
    return results


def print_validation_report(results: dict, logger=None):
    """Print validation report."""
    if logger is None:
        logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("Data Validation Report")
    logger.info("=" * 80)
    
    if results['valid']:
        logger.info("Status: VALID")
    else:
        logger.info("Status: ISSUES DETECTED")
    
    for split in ['train', 'val']:
        logger.info(f"\n{split.upper()}:")
        logger.info(f"  Images: {results[split]['images']}")
        logger.info(f"  Labels: {results[split]['labels']}")
    
    if results['issues']:
        logger.info("\nIssues:")
        for issue in results['issues']:
            logger.info(f"  - {issue}")
    
    logger.info("=" * 80)


def main():
    """Main entry point for data preparation."""
    parser = argparse.ArgumentParser(
        description='Prepare data for YOLOv8 tree detection'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Organize command
    organize_parser = subparsers.add_parser('organize', help='Organize data into train/val splits')
    organize_parser.add_argument('images', help='Path to directory with all images')
    organize_parser.add_argument('labels', help='Path to directory with all labels')
    organize_parser.add_argument('--output', default='data/processed', 
                                help='Output directory')
    organize_parser.add_argument('--train-ratio', type=float, default=0.8,
                                help='Ratio of data for training')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate YOLO format data')
    validate_parser.add_argument('data', help='Path to data directory')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    if args.command == 'organize':
        success = organize_yolo_format_data(
            args.images,
            args.labels,
            args.output,
            train_ratio=args.train_ratio,
            logger=logger
        )
        return 0 if success else 1
    
    elif args.command == 'validate':
        results = validate_yolo_format(args.data, logger)
        print_validation_report(results, logger)
        return 0 if results['valid'] else 1
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
