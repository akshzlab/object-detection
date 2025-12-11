"""Utility helpers for Sentinel-2 YOLO pipeline."""
import numpy as np
from typing import Tuple
from rasterio.warp import transform as rio_transform


def normalize_uint16_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize a uint16 Sentinel-2 band array to uint8 for visualization or YOLO input.

    Performs min-max normalization and clips to 0-255 range.

    Args:
        arr (np.ndarray): Input array (typically uint16 from Sentinel-2)

    Returns:
        np.ndarray: Normalized array as uint8

    Examples:
        >>> import numpy as np
        >>> sentinel_band = np.random.randint(0, 10000, (100, 100), dtype=np.uint16)
        >>> normalized = normalize_uint16_to_uint8(sentinel_band)
        >>> normalized.dtype
        dtype('uint8')
        >>> normalized.min(), normalized.max()
        (0, 255)
    """
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return (arr * 255).astype(np.uint8)


def reproject_coords(
    xs: np.ndarray,
    ys: np.ndarray,
    src_crs: str,
    dst_crs: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproject coordinate arrays between CRS systems.

    Args:
        xs (np.ndarray): X/longitude coordinates
        ys (np.ndarray): Y/latitude coordinates
        src_crs (str): Source CRS (e.g., 'EPSG:4326')
        dst_crs (str): Destination CRS (e.g., 'EPSG:3857')

    Returns:
        Tuple[np.ndarray, np.ndarray]: Reprojected (x, y) coordinates

    Examples:
        >>> import numpy as np
        >>> xs = np.array([0.0, 1.0, 2.0])
        >>> ys = np.array([50.0, 51.0, 52.0])
        >>> tx, ty = reproject_coords(xs, ys, 'EPSG:4326', 'EPSG:3857')
    """
    tx, ty = rio_transform(src_crs, dst_crs, xs, ys)
    return np.array(tx), np.array(ty)


def load_rgb_from_geotiff(
    src,
    r: int = 1,
    g: int = 2,
    b: int = 3
) -> np.ndarray:
    """Load 3 bands from an open rasterio dataset and return as HxWx3 array.

    Args:
        src: Open rasterio DatasetReader object
        r (int): Red band index (default: 1)
        g (int): Green band index (default: 2)
        b (int): Blue band index (default: 3)

    Returns:
        np.ndarray: RGB image array of shape (H, W, 3)

    Examples:
        >>> import rasterio
        >>> from sentinel2_yolo.utils import load_rgb_from_geotiff
        >>> with rasterio.open('scene.tif') as src:
        ...     rgb = load_rgb_from_geotiff(src, r=3, g=2, b=1)
        ...     rgb.shape
        (1024, 1024, 3)
    """
    img = src.read([r, g, b])
    return np.moveaxis(img, 0, -1)