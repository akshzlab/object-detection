"""
Prepare car detection dataset
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple
import random


def organize_images_and_labels(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[int, int, int]:
    """
    Organize images and labels into train/val/test split.
    
    Args:
        source_dir: Source directory with images and labels
        output_dir: Output directory for organized data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        
    Returns:
        Tuple of (train_count, val_count, test_count)
    """
    # Create output directories
    output_dirs = {
        'train': {
            'images': os.path.join(output_dir, 'train', 'images'),
            'labels': os.path.join(output_dir, 'train', 'labels')
        },
        'val': {
            'images': os.path.join(output_dir, 'val', 'images'),
            'labels': os.path.join(output_dir, 'val', 'labels')
        },
        'test': {
            'images': os.path.join(output_dir, 'test', 'images'),
            'labels': os.path.join(output_dir, 'test', 'labels')
        }
    }
    
    for split_dirs in output_dirs.values():
        for dir_path in split_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(source_dir).glob(f'*{ext}'))
        image_files.extend(Path(source_dir).glob(f'*{ext.upper()}'))
    
    image_files = sorted(list(set(image_files)))  # Remove duplicates and sort
    
    print(f"Found {len(image_files)} images")
    
    if not image_files:
        print("No images found in source directory")
        return 0, 0, 0
    
    # Shuffle images
    random.shuffle(image_files)
    
    # Calculate split indices
    train_idx = int(len(image_files) * train_ratio)
    val_idx = train_idx + int(len(image_files) * val_ratio)
    
    # Split images
    train_files = image_files[:train_idx]
    val_files = image_files[train_idx:val_idx]
    test_files = image_files[val_idx:]
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # Copy files
    counts = {}
    for split, files in splits.items():
        count = 0
        for image_file in files:
            # Copy image
            dest_image = os.path.join(output_dirs[split]['images'], 
                                     image_file.name)
            shutil.copy2(image_file, dest_image)
            
            # Copy label if exists
            label_file = image_file.with_suffix('.txt')
            if label_file.exists():
                dest_label = os.path.join(output_dirs[split]['labels'], 
                                        label_file.name)
                shutil.copy2(label_file, dest_label)
            
            count += 1
        
        counts[split] = count
        print(f"{split.capitalize()}: {count} images")
    
    return counts.get('train', 0), counts.get('val', 0), counts.get('test', 0)


def split_existing_data(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[int, int, int]:
    """
    Split existing images and labels into train/val/test.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        output_dir: Output directory for split data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        
    Returns:
        Tuple of (train_count, val_count, test_count)
    """
    # Create output directories
    output_dirs = {
        'train': {
            'images': os.path.join(output_dir, 'train', 'images'),
            'labels': os.path.join(output_dir, 'train', 'labels')
        },
        'val': {
            'images': os.path.join(output_dir, 'val', 'images'),
            'labels': os.path.join(output_dir, 'val', 'labels')
        },
        'test': {
            'images': os.path.join(output_dir, 'test', 'images'),
            'labels': os.path.join(output_dir, 'test', 'labels')
        }
    }
    
    for split_dirs in output_dirs.values():
        for dir_path in split_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
        image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
    
    image_files = sorted(list(set(image_files)))
    
    print(f"Found {len(image_files)} images")
    
    if not image_files:
        print("No images found")
        return 0, 0, 0
    
    # Shuffle
    random.shuffle(image_files)
    
    # Calculate split indices
    train_idx = int(len(image_files) * train_ratio)
    val_idx = train_idx + int(len(image_files) * val_ratio)
    
    # Split
    train_files = image_files[:train_idx]
    val_files = image_files[train_idx:val_idx]
    test_files = image_files[val_idx:]
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # Copy files
    counts = {}
    for split, files in splits.items():
        count = 0
        for image_file in files:
            # Copy image
            dest_image = os.path.join(output_dirs[split]['images'], 
                                     image_file.name)
            shutil.copy2(image_file, dest_image)
            
            # Copy label if exists
            label_file = Path(labels_dir) / image_file.with_suffix('.txt').name
            if label_file.exists():
                dest_label = os.path.join(output_dirs[split]['labels'], 
                                        label_file.name)
                shutil.copy2(label_file, dest_label)
            
            count += 1
        
        counts[split] = count
        print(f"{split.capitalize()}: {count} images")
    
    return counts.get('train', 0), counts.get('val', 0), counts.get('test', 0)


def validate_dataset(data_dir: str) -> bool:
    """
    Validate dataset structure and content.
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        True if valid, False otherwise
    """
    required_dirs = [
        'train/images', 'train/labels',
        'val/images', 'val/labels'
    ]
    
    print("Validating dataset structure...")
    
    for req_dir in required_dirs:
        path = os.path.join(data_dir, req_dir)
        if not os.path.isdir(path):
            print(f"Missing directory: {path}")
            return False
    
    # Check for matching files
    train_images = os.listdir(os.path.join(data_dir, 'train/images'))
    train_labels = os.listdir(os.path.join(data_dir, 'train/labels'))
    
    print(f"Train images: {len(train_images)}")
    print(f"Train labels: {len(train_labels)}")
    
    if not train_images:
        print("Warning: No training images found")
    
    if not train_labels:
        print("Warning: No training labels found")
    
    print("Dataset validation complete")
    return True


def main():
    """Main data preparation script"""
    parser = argparse.ArgumentParser(description='Prepare car detection dataset')
    
    parser.add_argument('--source', type=str,
                       help='Source directory with images and labels')
    parser.add_argument('--images', type=str,
                       help='Directory with images')
    parser.add_argument('--labels', type=str,
                       help='Directory with labels')
    parser.add_argument('--output', type=str, default='data',
                       help='Output directory for organized data')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio for training set')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Ratio for validation set')
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset after preparation')
    
    args = parser.parse_args()
    
    # Prepare data
    if args.source:
        if not os.path.exists(args.source):
            print(f"Error: Source directory not found: {args.source}")
            return
        
        print(f"Organizing images from: {args.source}")
        train, val, test = organize_images_and_labels(
            args.source, args.output, args.train_ratio, args.val_ratio
        )
    
    elif args.images and args.labels:
        if not os.path.exists(args.images):
            print(f"Error: Images directory not found: {args.images}")
            return
        
        if not os.path.exists(args.labels):
            print(f"Error: Labels directory not found: {args.labels}")
            return
        
        print(f"Splitting images from: {args.images}")
        train, val, test = split_existing_data(
            args.images, args.labels, args.output, args.train_ratio, args.val_ratio
        )
    
    else:
        parser.print_help()
        return
    
    print(f"\nDataset prepared in: {args.output}")
    print(f"Train: {train}, Val: {val}, Test: {test}")
    
    # Validate if requested
    if args.validate:
        validate_dataset(args.output)


if __name__ == '__main__':
    main()
