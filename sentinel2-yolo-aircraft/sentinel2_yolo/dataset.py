"""Dataset utilities for Sentinel-2 aircraft detection.
This module defines a dataset loader for YOLO training with proper preprocessing.
"""
import os
import logging
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class Sentinel2AircraftDataset(Dataset):
    """PyTorch Dataset for Sentinel-2 aircraft detection.

    Loads images and YOLO-format labels. Returns normalized tensors.

    Example:
        >>> dataset = Sentinel2AircraftDataset('/path/to/data')
        >>> image, labels = dataset[0]
        >>> image.shape
        torch.Size([3, 640, 640])
    """

    def __init__(
        self,
        root_dir: str,
        img_size: int = 640,
        augment: bool = True
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory containing images/ and labels/ subdirs
            img_size (int): Target image size for resize (default: 640)
            augment (bool): Apply data augmentation (default: True)
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.img_size = img_size
        self.augment = augment

        # Collect all image files recursively
        self.image_paths = []
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(self.image_dir, split)
            if os.path.isdir(split_dir):
                for f in os.listdir(split_dir):
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                        self.image_paths.append(os.path.join(split_dir, f))

        logger.info(f"Loaded {len(self.image_paths)} images from {root_dir}")

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        if augment:
            self.augment_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
            ])
        else:
            self.augment_transform = None

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load image and labels.

        Args:
            idx (int): Index

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (image tensor, labels tensor)
        """
        img_path = self.image_paths[idx]
        label_path = img_path.replace("images", "labels").rsplit('.', 1)[0] + ".txt"

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            raise

        # Apply augmentation if enabled
        if self.augment and self.augment_transform:
            image = self.augment_transform(image)

        # Apply standard transforms
        image = self.transform(image)

        # Load labels
        labels = []
        if os.path.exists(label_path):
            try:
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls, xc, yc, w, h = map(float, parts[:5])
                            labels.append([cls, xc, yc, w, h])
            except Exception as e:
                logger.warning(f"Failed to load labels from {label_path}: {e}")

        # Convert to tensor
        if labels:
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)

        return image, labels
