"""Training script wrapper for YOLO aircraft detection."""
import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO model for aircraft detection"
    )
    parser.add_argument(
        '--data',
        required=True,
        help='Path to data.yaml configuration file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--img',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=4,
        help='Batch size (default: 4)'
    )
    parser.add_argument(
        '--weights',
        default='yolov8n.pt',
        help='Initial weights to load (default: yolov8n.pt)'
    )
    parser.add_argument(
        '--device',
        default='0',
        help='Device to use for training (default: 0, use -1 for CPU)'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Validate data.yaml exists
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"data.yaml not found: {args.data}")
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    logger.info(f"Loading model from {args.weights}...")
    try:
        model = YOLO(args.weights)
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        raise

    logger.info(f"Starting training with config: {args.data}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Image size: {args.img}")
    logger.info(f"  Batch size: {args.batch}")
    logger.info(f"  Device: {args.device}")

    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.img,
            batch=args.batch,
            device=args.device,
            verbose=True
        )
        logger.info("✓ Training completed successfully")
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        raise


if __name__ == '__main__':
    main()