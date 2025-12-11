"""CLI: Run sliding-window inference on a GeoTIFF and export GeoJSON."""
import argparse
import logging
from sentinel2_yolo.inference import sliding_window_inference

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on a GeoTIFF with sliding-window tiling"
    )
    parser.add_argument(
        '--geotiff',
        required=True,
        help='Path to input GeoTIFF file'
    )
    parser.add_argument(
        '--weights',
        required=True,
        help='Path to YOLO model weights'
    )
    parser.add_argument(
        '--tile-size',
        type=int,
        default=1024,
        help='Tile size in pixels (default: 1024)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=200,
        help='Tile overlap in pixels (default: 200)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Confidence threshold for detections (default: 0.5)'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='Device to use (cpu or cuda, default: cpu)'
    )
    parser.add_argument(
        '--bands',
        type=int,
        nargs='+',
        default=[1, 2, 3],
        help='Band indices to read from GeoTIFF (default: 1 2 3)'
    )
    parser.add_argument(
        '--nms-iou',
        type=float,
        default=0.3,
        help='NMS IOU threshold (default: 0.3)'
    )
    parser.add_argument(
        '--out',
        default='data/detections/scene.geojson',
        help='Output GeoJSON file path (default: data/detections/scene.geojson)'
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger.info(f"Running inference with parameters:")
    logger.info(f"  GeoTIFF: {args.geotiff}")
    logger.info(f"  Weights: {args.weights}")
    logger.info(f"  Confidence threshold: {args.conf}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Band indices: {args.bands}")

    sliding_window_inference(
        args.geotiff,
        args.weights,
        tile_size_px=args.tile_size,
        overlap=args.overlap,
        out_geojson=args.out,
        conf_threshold=args.conf,
        device=args.device,
        band_indices=args.bands,
        nms_iou=args.nms_iou,
    )


if __name__ == '__main__':
    main()