"""Sliding-window inference for large Sentinel-2 GeoTIFFs.
Exports detections as GeoJSON with deduplication.
"""
import logging
from typing import List, Optional
import rasterio
import numpy as np
from shapely.geometry import box, mapping
from geopandas import GeoDataFrame
from ultralytics import YOLO
from .tiling import tile_image

logger = logging.getLogger(__name__)


def sliding_window_inference(
    geotiff_path: str,
    weights: str,
    tile_size_px: int = 1024,
    overlap: int = 200,
    out_geojson: str = "detections.geojson",
    conf_threshold: float = 0.5,
    device: str = 'cpu',
    band_indices: Optional[List[int]] = None,
    nms_iou: float = 0.3
) -> str:
    """Run YOLO inference on a large GeoTIFF using sliding-window tiling.

    Applies confidence filtering and NMS deduplication for overlapping tiles.

    Args:
        geotiff_path (str): Path to GeoTIFF file
        weights (str): Path to YOLO model weights
        tile_size_px (int): Tile size in pixels (default: 1024)
        overlap (int): Sliding window overlap in pixels (default: 200)
        out_geojson (str): Output GeoJSON file path (default: detections.geojson)
        conf_threshold (float): Confidence threshold for filtering detections (default: 0.5)
        device (str): Device to run inference on ('cpu' or 'cuda', default: 'cpu')
        band_indices (Optional[List[int]]): Band indices to read from GeoTIFF (default: [1,2,3])
        nms_iou (float): IOU threshold for Non-Maximum Suppression (default: 0.3)

    Returns:
        str: Path to output GeoJSON file

    Examples:
        >>> from sentinel2_yolo.inference import sliding_window_inference
        >>> sliding_window_inference(
        ...     'scene.tif',
        ...     'best.pt',
        ...     conf_threshold=0.5
        ... )
        'detections.geojson'
    """
    if band_indices is None:
        band_indices = [1, 2, 3]

    logging.basicConfig(level=logging.INFO)

    logger.info(f"Loading model from {weights}...")
    model = YOLO(weights)
    model.to(device)

    logger.info(f"Reading GeoTIFF: {geotiff_path}")
    with rasterio.open(geotiff_path) as src:
        if src.count < max(band_indices):
            raise ValueError(
                f"GeoTIFF has {src.count} bands but requested bands {band_indices}"
            )
        img = src.read(band_indices)
        img = np.moveaxis(img, 0, -1)

        transform = src.transform
        crs = src.crs

    detections = []
    logger.info(f"Tiling image with {tile_size_px}px tiles and {overlap}px overlap...")

    tiles = tile_image(img, tile_size_px, overlap)
    logger.info(f"Processing {len(tiles)} tiles...")

    for tile_idx, (tile, x, y) in enumerate(tiles, 1):
        results = model.predict(tile, verbose=False, device=device)[0]

        h, w = tile.shape[:2]
        for box_idx, box_det in enumerate(results.boxes.xyxy.cpu().numpy()):
            conf = float(results.boxes.conf[box_idx].cpu().numpy())

            # Filter by confidence threshold
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = box_det

            # Shift to global pixel coordinates
            gx1 = x1 + x
            gy1 = y1 + y
            gx2 = x2 + x
            gy2 = y2 + y

            # Convert to map coordinates
            lon1, lat1 = rasterio.transform.xy(transform, gy1, gx1)
            lon2, lat2 = rasterio.transform.xy(transform, gy2, gx2)

            geom = box(lon1, lat2, lon2, lat1)
            class_id = int(results.boxes.cls[box_idx].cpu().numpy())
            class_name = model.names[class_id]

            detections.append({
                "geometry": mapping(geom),
                "properties": {
                    "class": class_name,
                    "confidence": conf,
                    "class_id": class_id
                }
            })

        if tile_idx % max(1, len(tiles) // 10) == 0:
            logger.info(f"  Processed {tile_idx}/{len(tiles)} tiles, {len(detections)} detections found")

    logger.info(f"Creating GeoDataFrame with {len(detections)} detections...")
    gdf = GeoDataFrame.from_features(detections, crs=crs)

    # Apply spatial deduplication (NMS)
    if len(gdf) > 0:
        logger.info(f"Applying NMS with IOU threshold {nms_iou}...")
        gdf = _apply_nms(gdf, nms_iou)
        logger.info(f"  After NMS: {len(gdf)} detections")

    logger.info(f"Saving detections to {out_geojson}...")
    gdf.to_file(out_geojson, driver="GeoJSON")
    logger.info(f"✓ Successfully saved {len(gdf)} detections to {out_geojson}")

    return out_geojson


def _apply_nms(gdf: GeoDataFrame, iou_threshold: float = 0.3) -> GeoDataFrame:
    """Apply Non-Maximum Suppression to remove duplicate detections.

    Removes duplicates from overlapping tiles using IOU-based suppression.

    Args:
        gdf (GeoDataFrame): GeoDataFrame with detection geometries
        iou_threshold (float): IOU threshold for suppression (default: 0.3)

    Returns:
        GeoDataFrame: Deduplicated GeoDataFrame
    """
    if len(gdf) == 0:
        return gdf

    # Sort by confidence descending
    gdf = gdf.sort_values('confidence', ascending=False).reset_index(drop=True)

    keep_indices = list(range(len(gdf)))
    for i in range(len(gdf)):
        if i not in keep_indices:
            continue

        # Compare with remaining boxes
        box_i = gdf.geometry[i]
        for j in range(i + 1, len(gdf)):
            if j not in keep_indices:
                continue

            box_j = gdf.geometry[j]
            intersection = box_i.intersection(box_j).area
            union = box_i.union(box_j).area
            iou = intersection / union if union > 0 else 0

            if iou > iou_threshold:
                keep_indices.remove(j)

    return gdf.iloc[keep_indices].reset_index(drop=True)
