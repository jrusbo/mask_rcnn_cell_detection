"""
Common tiling functions for image processing.

WARNING:
    These are independent of any ML framework (e.g., NO TORCH OR TF imports).
"""

import math


def get_grid_coords(
    tile_size: int, overlap: float, grid_x: int, grid_y: int, grid_center: tuple[int, int] = None
) -> tuple[int, int, int, int]:
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
    tile_size: int, overlap: float, image_width: int, image_height: int
) -> list[tuple[int, int, int, int]]:
    tile_distance = tile_size * (1 - overlap)

    tile_count_x = 1 if image_width <= tile_size else 1 + math.ceil((image_width - tile_size) / tile_distance)
    tile_count_y = 1 if image_height <= tile_size else 1 + math.ceil((image_height - tile_size) / tile_distance)

    grid_center = (math.ceil(tile_size / 2), math.ceil(tile_size / 2))

    tile_coords = []
    for tile_x in range(0, tile_count_x):
        for tile_y in range(0, tile_count_y):
            x_min, x_max, y_min, y_max = get_grid_coords(tile_size, overlap, tile_x, tile_y, grid_center)

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
    tile_coords.sort(key=lambda x: (x[2] + x[0]))
    return tile_coords



