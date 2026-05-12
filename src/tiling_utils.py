"""Framework-agnostic helpers for generating overlapping image tiles."""

import math


def get_grid_coords(
    tile_size: int,
    overlap: float,
    grid_x: int,
    grid_y: int,
    grid_center: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Return the coordinates for a tile in a regular overlapping grid.

    Computes the bounding rectangle for a single tile within a grid of overlapping tiles,
    optionally centered at a specified point.

    Args:
        tile_size: Size of each tile in pixels (assumes square tiles).
        overlap: Fraction of overlap between adjacent tiles (between 0 and 1).
            For example, 0.25 means 25% overlap.
        grid_x: Column index of the tile in the grid (0-based).
        grid_y: Row index of the tile in the grid (0-based).
        grid_center: Optional (x, y) coordinate around which to center the grid.
            If None, defaults to (0, 0). Defaults to None.

    Returns:
        Tuple of (x_min, x_max, y_min, y_max) defining the tile's bounding box.
    """

    ts_2 = tile_size // 2
    shift_right = grid_x * tile_size * (1 - overlap)
    shift_down = grid_y * tile_size * (1 - overlap)
    grid_center_x, grid_center_y = grid_center if grid_center is not None else (0, 0)

    x_min = int(-ts_2 + shift_right + grid_center_x)
    x_max = x_min + tile_size
    y_min = int(-ts_2 + shift_down + grid_center_y)
    y_max = y_min + tile_size

    return x_min, x_max, y_min, y_max


def get_tile_coords_for_full_image(
    tile_size: int,
    overlap: float,
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int]]:
    """Generate a unique, sorted set of tile coordinates that cover an image.

    Computes a set of non-redundant tile coordinates that completely cover an image
    with the specified tile size and overlap. The last row and column of tiles are
    adjusted to align with the image boundaries.

    Args:
        tile_size: Size of each tile in pixels (assumes square tiles).
        overlap: Fraction of overlap between adjacent tiles (between 0 and 1).
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        List of tuples (x_min, x_max, y_min, y_max) sorted by row then column.
        Each tuple defines a non-overlapping bounding box that covers part of the image.
    """

    tile_distance = tile_size * (1 - overlap)

    tile_count_x = (
        1
        if image_width <= tile_size
        else 1 + math.ceil((image_width - tile_size) / tile_distance)
    )
    tile_count_y = (
        1
        if image_height <= tile_size
        else 1 + math.ceil((image_height - tile_size) / tile_distance)
    )

    grid_center = (math.ceil(tile_size / 2), math.ceil(tile_size / 2))

    tile_coords = []
    for tile_x in range(0, tile_count_x):
        for tile_y in range(0, tile_count_y):
            x_min, x_max, y_min, y_max = get_grid_coords(
                tile_size, overlap, tile_x, tile_y, grid_center
            )

            if tile_count_x == 1 and image_width <= tile_size:
                x_min, x_max = 0, image_width
            elif tile_x == tile_count_x - 1:
                x_min = image_width - tile_size
                x_max = image_width

            if tile_count_y == 1 and image_height <= tile_size:
                y_min, y_max = 0, image_height
            elif tile_y == tile_count_y - 1:
                y_min = image_height - tile_size
                y_max = image_height

            tile_coords.append((x_min, x_max, y_min, y_max))

    tile_coords = list(set(tile_coords))
    tile_coords.sort(key=lambda x: x[2] + x[0])
    return tile_coords
