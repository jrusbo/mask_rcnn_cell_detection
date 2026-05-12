"""Convert raw cell masks into a COCO-style tiled dataset."""

import json
import os
import shutil
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from skimage import io as sio
from tqdm import tqdm

from tiling_utils import get_tile_coords_for_full_image


RAW_DATA_DIR = r"/kaggle/input/datasets/quinonycu/hw3-maskrcnn/train"
OUT_DIR = "datasets/coco_format"
USE_TILING = True  # Set to False to support full image without tiling, beware of memory issues with large images.
PATCH_SIZE = 512  # Size of the square tiles to create (if tiling is enabled)
OVERLAP = 0.25  # overlap between adjacent tiles
KERNEL_SIZE = (
    1,
    1,
)  # Kernel size for morphological closing. Adjust based on expected cell size and noise level.
MIN_AREA = 20  # Minimum area for a valid instance to be included in annotations. Adjust based on expected cell size.
RANDOM_SEED = 42
SPLIT_RATIO = 0.8
NUM_CLASSES = 4

img_id = 1
ann_id = 1
images_info: list[dict[str, Any]] = []

os.makedirs(os.path.join(OUT_DIR, "images/train"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "images/val"), exist_ok=True)


def _clear_output_dir(path: str) -> None:
    """Remove all files and folders inside `path` if it exists.

    Args:
        path: Directory path to clear. If the path does not exist or is not a directory,
            this function does nothing.
    """

    if os.path.isdir(path):
        for entry in os.scandir(path):
            if entry.is_file():
                os.remove(entry.path)
            elif entry.is_dir():
                shutil.rmtree(entry.path)


def encode_mask(binary_mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask using the COCO run-length encoding format.

    Converts a boolean or binary numpy array into the COCO RLE (run-length encoding)
    format, which is a space-efficient representation used by the pycocotools library.

    Args:
        binary_mask: A 2D numpy array with binary values (0 or 1, or bool type)
            representing the mask.

    Returns:
        A dictionary with RLE encoding containing 'counts' and 'size' keys.
        The 'counts' value is a UTF-8 decoded string suitable for JSON serialization.
    """

    arr = np.asarray(binary_mask, dtype=np.uint8, order="F")
    rle = mask_utils.encode(arr)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def read_maskfile(filepath: str) -> np.ndarray:
    """Load a TIFF mask with skimage so multi-page TIFFs are handled safely.

    Uses scikit-image to read TIFF files, which better handles multi-page TIFFs
    and other TIFF variants compared to OpenCV.

    Args:
        filepath: Path to the TIFF mask file.

    Returns:
        A 2D numpy array representing the mask. If the file contains multiple
        channels, only the first channel is returned.
    """

    mask_array = sio.imread(filepath)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    return mask_array


def _ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Normalize an image into an RGB `uint8` array suitable for OpenCV.

    Handles multiple input formats:
    - Grayscale (2D) -> converted to RGB by replicating channels
    - Single-channel or 4-channel -> converted to 3-channel RGB
    - Non-uint8 types -> normalized to 0-255 range using percentile clipping

    Args:
        img: Input image as a numpy array in any format (grayscale, RGB, RGBA, etc.).

    Returns:
        A 3-channel, uint8 numpy array in RGB format with shape (height, width, 3).
    """

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[-1]:
        img = np.transpose(img, (1, 2, 0))

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        finite_mask = np.isfinite(img)
        if not np.any(finite_mask):
            return np.zeros_like(img, dtype=np.uint8)

        finite_vals = img[finite_mask]
        lo = float(np.percentile(finite_vals, 1))
        hi = float(np.percentile(finite_vals, 99))
        if hi <= lo:
            lo = float(finite_vals.min())
            hi = float(finite_vals.max())

        if hi > lo:
            img = (img - lo) * (255.0 / (hi - lo))
        else:
            img = img - lo

        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def process_sample(
    sample_folder: str, is_val: bool = False
) -> tuple[list[dict[str, Any]], int]:
    """Tile a single sample folder and return its COCO annotations and empty-tile count.

    Processes a single sample folder by:
    1. Reading the image and instance masks
    2. Tiling the image into patches (if enabled)
    3. Extracting per-instance annotations
    4. Encoding masks in COCO RLE format

    Args:
        sample_folder: Path to a sample folder containing 'image.tif' and 'class*.tif' files.
        is_val: Whether this sample belongs to the validation split. Defaults to False.

    Returns:
        Tuple of (annotations_list, empty_tiles_count) where:
        - annotations_list is a list of COCO annotation dicts
        - empty_tiles_count is the number of tiles with no instances
    """

    global img_id, ann_id
    split_str = "val" if is_val else "train"

    img_path = os.path.join(sample_folder, "image.tif")
    if not os.path.exists(img_path):
        print(f"Warning: Image {img_path} does not exist!")
        return [], 0

    try:
        img = sio.imread(img_path)
    except Exception as e:
        print(f"Warning: Failed to read {img_path}: {e}")
        return [], 0

    if img.size == 0:
        print(f"Warning: Image {img_path} is empty!")
        return [], 0

    img = _ensure_uint8_rgb(img)

    h_img, w_img = img.shape[:2]
    # Morphological kernel:
    # - Large enough to fill small holes
    # - Small enough to preserve fine cell boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_SIZE)
    annotations = []
    empty_tiles_count = 0

    if USE_TILING:
        tiles = get_tile_coords_for_full_image(PATCH_SIZE, OVERLAP, w_img, h_img)
    else:
        tiles = [(0, w_img, 0, h_img)]

    for x_min, x_max, y_min, y_max in tiles:
        cropped = img[y_min:y_max, x_min:x_max]
        actual_h, actual_w = cropped.shape[:2]
        if actual_h == 0 or actual_w == 0:
            continue

        pad_h = PATCH_SIZE - actual_h
        pad_w = PATCH_SIZE - actual_w

        top_pad = pad_h // 2
        bottom_pad = pad_h - top_pad
        left_pad = pad_w // 2
        right_pad = pad_w - left_pad

        if pad_h > 0 or pad_w > 0:
            cropped = cv2.copyMakeBorder(
                cropped,
                top_pad,
                bottom_pad,
                left_pad,
                right_pad,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
            actual_h, actual_w = cropped.shape[:2]

        patch_filename = f"{os.path.basename(sample_folder)}_{x_min}_{y_min}.jpg"
        cv2.imwrite(
            os.path.join(OUT_DIR, f"images/{split_str}", patch_filename),
            cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR),
        )

        patch_anns = []
        for class_id in range(1, NUM_CLASSES + 1):
            mask_path = os.path.join(sample_folder, f"class{class_id}.tif")
            if not os.path.exists(mask_path):
                continue

            try:
                mask_map = read_maskfile(mask_path)
            except Exception as e:
                print(f"Warning: Failed to read {mask_path}: {e}")
                continue

            if mask_map.size == 0:
                continue

            if mask_map.dtype in (np.float64, np.float32):
                mask_map = mask_map.astype(np.int32)
            elif (
                mask_map.dtype != np.uint8
                and mask_map.dtype != np.uint16
                and mask_map.dtype != np.int32
            ):
                mask_map = (
                    mask_map.astype(np.uint16)
                    if mask_map.max() > 255
                    else mask_map.astype(np.uint8)
                )

            patch_mask_map = mask_map[y_min:y_max, x_min:x_max]

            if pad_h > 0 or pad_w > 0:
                patch_mask_map = cv2.copyMakeBorder(
                    patch_mask_map,
                    top_pad,
                    bottom_pad,
                    left_pad,
                    right_pad,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )

            for inst_id in np.unique(patch_mask_map):
                if inst_id == 0:
                    continue

                binary_mask = np.uint8(patch_mask_map == inst_id)
                closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                full_patch_mask = closed_mask

                px, py, pw, ph = cv2.boundingRect(full_patch_mask)
                if pw * ph < MIN_AREA:
                    continue

                rle = encode_mask(full_patch_mask)
                area_value = int(mask_utils.area(rle))

                patch_anns.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": class_id,
                        "bbox": [int(px), int(py), int(pw), int(ph)],
                        "area": area_value,
                        "segmentation": rle,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        images_info.append(
            {
                "id": img_id,
                "width": actual_w,
                "height": actual_h,
                "file_name": patch_filename,
            }
        )
        img_id += 1

        if len(patch_anns) > 0:
            annotations.extend(patch_anns)
        else:
            empty_tiles_count += 1

    return annotations, empty_tiles_count


def main() -> None:
    """Run the full preprocessing pipeline and write COCO JSON files to disk.

    Orchestrates the entire dataset preprocessing workflow:
    1. Collects and splits sample folders into train/val sets
    2. Processes each sample (tiling, mask extraction, RLE encoding)
    3. Writes train.json and val.json in COCO format
    4. Prints dataset statistics (number of images, instances per class)

    Reads from RAW_DATA_DIR and writes to OUT_DIR with paths defined as module constants.
    """

    global img_id, ann_id, images_info

    folders = [f.path for f in os.scandir(RAW_DATA_DIR) if f.is_dir()]
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(folders)
    split_idx = int(len(folders) * SPLIT_RATIO)

    categories = [{"id": i, "name": f"cell_{i}"} for i in range(1, NUM_CLASSES + 1)]

    global_stats = {"train": {}, "val": {}}
    for split, folder_subset in [
        ("train", folders[:split_idx]),
        ("val", folders[split_idx:]),
    ]:
        img_id, ann_id = 1, 1
        images_info, all_annotations = [], []
        empty_tiles_total = 0

        _clear_output_dir(os.path.join(OUT_DIR, f"images/{split}"))

        print(f"\nProcessing {split} split...")
        for folder in tqdm(folder_subset, desc=f"Extracting & Tiling ({split})"):
            anns, empty_count = process_sample(folder, is_val=(split == "val"))
            all_annotations.extend(anns)
            empty_tiles_total += empty_count

        with open(os.path.join(OUT_DIR, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "images": images_info,
                    "annotations": all_annotations,
                    "categories": categories,
                },
                f,
            )

        print(
            f"Saved {len(images_info)} images and {len(all_annotations)} annotations for {split} split."
        )

        class_counts = {c: 0 for c in range(1, NUM_CLASSES + 1)}
        for ann in all_annotations:
            class_counts[ann["category_id"]] += 1
        global_stats[split] = {
            "images": len(images_info),
            "empty_tiles": empty_tiles_total,
            "annotations": len(all_annotations),
            "classes": class_counts,
        }

    print("\n==================================")
    print("       DATASET STATISTICS         ")
    print("==================================")
    for split in ["train", "val"]:
        print(f"\n[{split.upper()} SPLIT]")
        print(f"  Total Patches Kept : {global_stats[split]['images']}")
        print(f"  Empty Patches      : {global_stats[split]['empty_tiles']}")
        print(f"  Total Instances    : {global_stats[split]['annotations']}")
        for c_id, count in global_stats[split]["classes"].items():
            print(f"    - Class {c_id}: {count}")
    print("==================================\n")

    print("Data processing successfully completed!")


if __name__ == "__main__":
    main()
