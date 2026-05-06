import os
import json
import shutil
import numpy as np
import cv2
from skimage import io as sio
from pycocotools import mask as mask_utils
from tqdm import tqdm

from tiling_utils import get_tile_coords_for_full_image


RAW_DATA_DIR = "/kaggle/input/datasets/quinonycu/hw3-maskrcnn/train" # "datasets/train"
OUT_DIR = "datasets/coco_format"
USE_TILING = True  # Set to False to support full image without tiling
PATCH_SIZE = 512
OVERLAP = 0.25  # 25% overlap between adjacent tiles
THRESHOLD = 0
KERNEL_SIZE = (1, 1)
MIN_AREA = 20
RANDOM_SEED = 42
SPLIT_RATIO = 0.8
NUM_CLASSES = 4

os.makedirs(os.path.join(OUT_DIR, "images/train"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "images/val"), exist_ok=True)


def _clear_output_dir(path: str):
    if os.path.isdir(path):
        for entry in os.scandir(path):
            if entry.is_file():
                os.remove(entry.path)
            elif entry.is_dir():
                shutil.rmtree(entry.path)


# --- User Provided Helpers ---

def encode_mask(binary_mask):
    # This creates the "weird string" - it is the official COCO RLE format!
    arr = np.asarray(binary_mask, dtype=np.uint8, order="F")
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def read_maskfile(filepath):
    # Read mask file with skimage to avoid OpenCV TIFF warnings
    mask_array = sio.imread(filepath)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    return mask_array


def _ensure_uint8_rgb(img):
    # Normalize layout first.
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



# --- Processing Logic ---
def process_sample(sample_folder, is_val=False):
    global img_id, ann_id
    split_str = "val" if is_val else "train"

    img_path = os.path.join(sample_folder, 'image.tif')
    if not os.path.exists(img_path):
        print(f"Warning: Image {img_path} does not exist!")
        return [], 0

    # Read image using skimage - it handles TIFF properly without warnings
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
    # Morphological kernel: 7x7 ellipse is good balance for small cells
    # - Large enough to fill small holes (5-10 px)
    # - Small enough to preserve fine cell boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_SIZE)
    annotations = []
    empty_tiles_count = 0

    if USE_TILING:
        tiles = get_tile_coords_for_full_image(PATCH_SIZE, OVERLAP, w_img, h_img)
    else:
        tiles = [(0, w_img, 0, h_img)]

    for (x_min, x_max, y_min, y_max) in tiles:
        cropped = img[y_min:y_max, x_min:x_max]
        actual_h, actual_w = cropped.shape[:2]
        if actual_h == 0 or actual_w == 0:
            continue

        patch_filename = f"{os.path.basename(sample_folder)}_{x_min}_{y_min}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, f"images/{split_str}", patch_filename),
                    cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

        patch_anns = []
        for class_id in range(1, NUM_CLASSES + 1):
            mask_path = os.path.join(sample_folder, f'class{class_id}.tif')
            if not os.path.exists(mask_path): continue

            try:
                mask_map = read_maskfile(mask_path)
            except Exception as e:
                print(f"Warning: Failed to read {mask_path}: {e}")
                continue

            if mask_map.size == 0:
                continue

            # Ensure mask is in integer format for instance ID comparison
            if mask_map.dtype in (np.float64, np.float32):
                mask_map = mask_map.astype(np.int32)
            elif mask_map.dtype != np.uint8 and mask_map.dtype != np.uint16 and mask_map.dtype != np.int32:
                mask_map = mask_map.astype(np.uint16) if mask_map.max() > 255 else mask_map.astype(np.uint8)

            patch_mask_map = mask_map[y_min:y_max, x_min:x_max]

            # In this dataset, instances might be encoded as unique float numbers
            for inst_id in np.unique(patch_mask_map):
                if inst_id == 0: continue

                # Binary instance mask
                binary_mask = np.uint8(patch_mask_map == inst_id)
                # Morphological close: removes small holes and connects nearby fragments
                closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                full_patch_mask = closed_mask

                px, py, pw, ph = cv2.boundingRect(full_patch_mask)
                if pw * ph < MIN_AREA: continue

                # Always store segmentation as COCO RLE (required for downstream training/consumers)
                rle = encode_mask(full_patch_mask)
                area_value = int(mask_utils.area(rle))

                patch_anns.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": [int(px), int(py), int(pw), int(ph)],
                    "area": area_value,
                    "segmentation": rle,
                    "iscrowd": 0
                })
                ann_id += 1

        images_info.append({
            "id": img_id,
            "width": actual_w,
            "height": actual_h,
            "file_name": patch_filename
        })
        img_id += 1

        if len(patch_anns) > 0:
            annotations.extend(patch_anns)
        else:
            empty_tiles_count += 1

    return annotations, empty_tiles_count


# --- Execution ---
folders = [f.path for f in os.scandir(RAW_DATA_DIR) if f.is_dir()]
np.random.seed(RANDOM_SEED)
np.random.shuffle(folders)
split_idx = int(len(folders) * SPLIT_RATIO)

categories = [{"id": i, "name": f"cell_{i}"} for i in range(1, NUM_CLASSES + 1)]

# Main Loop with TQDM Progress Bar
global_stats = {"train": {}, "val": {}}
for split, folder_subset in [("train", folders[:split_idx]), ("val", folders[split_idx:])]:
    img_id, ann_id = 1, 1
    images_info, all_annotations = [], []
    empty_tiles_total = 0

    _clear_output_dir(os.path.join(OUT_DIR, f"images/{split}"))

    print(f"\nProcessing {split} split...")
    for folder in tqdm(folder_subset, desc=f"Extracting & Tiling ({split})"):
        anns, empty_count = process_sample(folder, is_val=(split == "val"))
        all_annotations.extend(anns)
        empty_tiles_total += empty_count

    with open(os.path.join(OUT_DIR, f"{split}.json"), "w") as f:
        json.dump({"images": images_info, "annotations": all_annotations, "categories": categories}, f)

    print(f"Saved {len(images_info)} images and {len(all_annotations)} annotations for {split} split.")

    # Store stats
    class_counts = {c: 0 for c in range(1, NUM_CLASSES + 1)}
    for ann in all_annotations:
        class_counts[ann["category_id"]] += 1
    global_stats[split] = {
        "images": len(images_info),
        "empty_tiles": empty_tiles_total,
        "annotations": len(all_annotations),
        "classes": class_counts
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
