"""Tiling utilities for sliding-window inference on large GeoTIFFs."""
import numpy as np
from typing import List, Tuple


def tile_image(
    image: np.ndarray,
    tile_size: int,
    overlap: int,
    pad_mode: str = 'reflect'
) -> List[Tuple[np.ndarray, int, int]]:
    """Divide a large image array into overlapping tiles.

    Uses reflection padding at edges to avoid artifacts from zero-padding.

    Args:
        image (np.ndarray): HxWxC image array
        tile_size (int): Size of each square tile in pixels
        overlap (int): Number of pixels to overlap between tiles
        pad_mode (str): Padding mode ('reflect', 'edge', 'constant', default: 'reflect')

    Returns:
        List[Tuple[np.ndarray, int, int]]: List of (tile, x, y) tuples
            tile = tile image array (tile_size x tile_size x C)
            x, y = top-left corner coordinates in original image

    Examples:
        >>> import numpy as np
        >>> img = np.random.rand(1024, 1024, 3)
        >>> tiles = tile_image(img, 512, 100)
        >>> len(tiles)  # 9 tiles for overlapping grid
        9
    """
    H, W = image.shape[:2]
    stride = tile_size - overlap

    tiles = []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            # Extract tile region
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            tile = image[y:y_end, x:x_end]

            # Pad tile if at image boundaries
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                pad_height = tile_size - tile.shape[0]
                pad_width = tile_size - tile.shape[1]

                if pad_mode == 'reflect':
                    # Reflection padding
                    pad_width_tuple = ((0, pad_height), (0, pad_width), (0, 0))
                    tile = np.pad(tile, pad_width_tuple, mode='reflect')
                elif pad_mode == 'edge':
                    # Edge padding
                    pad_width_tuple = ((0, pad_height), (0, pad_width), (0, 0))
                    tile = np.pad(tile, pad_width_tuple, mode='edge')
                else:
                    # Zero padding (original behavior)
                    padded = np.zeros((tile_size, tile_size, image.shape[2]), dtype=image.dtype)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded

            tiles.append((tile, x, y))

    return tiles
