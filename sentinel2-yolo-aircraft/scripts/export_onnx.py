"""Export YOLO model to ONNX format for deployment."""
import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLO model to ONNX format"
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to YOLOv8 .pt model weights"
    )
    parser.add_argument(
        "--out",
        default="models/aircraft_model.onnx",
        help="Output path for ONNX model"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size for the model"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version"
    )
    args = parser.parse_args()

    # Validate input file
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {args.weights}")

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Loading model from {args.weights}...")

    try:
        model = YOLO(args.weights)
        logger.info(f"Exporting to ONNX format...")
        output_path = model.export(
            format='onnx',
            imgsz=args.img_size,
            opset=args.opset
        )
        logger.info(f"✓ Model exported successfully to: {output_path}")
    except Exception as e:
        logger.error(f"✗ Export failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Path to YOLOv8 .pt model')
    parser.add_argument('--out', default='models/aircraft_model.onnx')
    parser.add_argument('--img-size', type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.export(format='onnx', imgsz=args.img_size, opset=13)


if __name__ == '__main__':
    main()