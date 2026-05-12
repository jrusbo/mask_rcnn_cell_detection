"""Visual sanity-check helpers for generated COCO annotations."""

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils


ROOT = Path(__file__).resolve().parent.parent
COCO_DIR = ROOT / "datasets" / "coco_format"
ANNOTATION_FILES = [COCO_DIR / "train.json", COCO_DIR / "val.json"]


def resolve_image_path(file_name: str) -> Path:
    """Find a generated image file in the known dataset locations.

    Searches for an image file in multiple standard locations within the COCO dataset
    directory, returning the first match found.

    Args:
        file_name: Name of the image file to locate (without directory path).

    Returns:
        Path object pointing to the found image file. Returns the first candidate
        path even if the file does not exist.
    """

    candidates = [
        COCO_DIR / "images" / "train" / file_name,
        COCO_DIR / "images" / "val" / file_name,
        COCO_DIR / "images" / file_name,
        COCO_DIR / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


annotation_file = next((path for path in ANNOTATION_FILES if path.exists()), None)
if annotation_file is None:
    raise FileNotFoundError(
        f"No COCO annotations found in: {', '.join(str(p) for p in ANNOTATION_FILES)}"
    )

with annotation_file.open("r", encoding="utf-8") as f:
    coco_data = json.load(f)

if not coco_data.get("images"):
    raise ValueError(f"No images found in annotation file: {annotation_file}")

img_info = random.choice(coco_data["images"])
img_path = resolve_image_path(img_info["file_name"])

img = cv2.imread(str(img_path))
if img is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")

anns = [a for a in coco_data["annotations"] if a["image_id"] == img_info["id"]]
categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
alpha = 0.35


def mask_color(mask_index: int) -> tuple[int, int, int]:
    """Generate a deterministic color from a mask index.

    Maps a mask index to a BGR color value using HUE interpolation in HSV space.
    The same index always produces the same color.

    Args:
        mask_index: Integer index of the mask (typically annotation index within an image).

    Returns:
        Tuple of (B, G, R) values suitable for OpenCV drawing functions.
    """

    hue = (mask_index * 37) % 180
    color = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def class_color(category_id: int) -> tuple[int, int, int]:
    """Generate a deterministic color from a category id.

    Maps a COCO category id to a BGR color value using HUE interpolation in HSV space.
    The same category always produces the same color.

    Args:
        category_id: COCO category ID (typically 1-based for foreground classes).

    Returns:
        Tuple of (B, G, R) values suitable for OpenCV drawing functions.
    """

    hue = ((category_id - 1) * 37) % 180
    color = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_segmentation(
    target_img: np.ndarray, annotation: dict[str, Any], color: tuple[int, int, int]
) -> bool:
    """Draw one COCO segmentation annotation onto an image in-place.

    Supports both RLE (run-length encoded) and polygon segmentation formats.
    Modifies target_img in-place, setting pixels in the segmentation mask to the given color.

    Args:
        target_img: Target image array with shape (height, width, 3) in BGR format.
            Will be modified in-place.
        annotation: COCO annotation dictionary containing a 'segmentation' key with
            either RLE dict or list of polygon coordinates.
        color: BGR color tuple (B, G, R) to draw the segmentation with.

    Returns:
        True if segmentation was successfully drawn, False if segmentation was missing
        or invalid.
    """

    segmentation = annotation.get("segmentation")
    if not segmentation:
        return False

    if isinstance(segmentation, dict):
        # RLE
        try:
            mask = mask_utils.decode(segmentation).astype(bool)  # type: ignore[arg-type]
        except Exception:
            return False
        target_img[mask] = color
        return True

    if isinstance(segmentation, list):
        for polygon in segmentation:
            if len(polygon) < 6:
                continue
            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(target_img, [pts], color)
        return True

    return False


def main() -> None:
    """Create preview overlays for a random sample from the COCO dataset.

    Loads a random image and its annotations from the COCO dataset, then generates
    and saves three preview images:
    - Original image
    - Image with masks overlaid with per-instance colors
    - Image with masks overlaid with per-class colors and bounding boxes labeled

    Saves PNGs to the COCO directory and attempts to display the annotated result.
    """

    orig_img = img.copy()
    overlay = img.copy()
    for idx, ann in enumerate(anns):
        draw_segmentation(overlay, ann, mask_color(idx))

    overlay_blended = cv2.addWeighted(overlay, alpha, orig_img, 1 - alpha, 0)

    annotated_overlay = img.copy()
    for ann in anns:
        draw_segmentation(annotated_overlay, ann, class_color(ann["category_id"]))

    annotated_blended = cv2.addWeighted(
        annotated_overlay, alpha, orig_img, 1 - alpha, 0
    )

    for ann in anns:
        x, y, w, h = ann["bbox"]
        cat_id = ann["category_id"]
        color = class_color(cat_id)
        cat_name = categories.get(cat_id, f"class_{cat_id}")
        cv2.rectangle(
            annotated_blended, (int(x), int(y)), (int(x + w), int(y + h)), color, 2
        )
        cv2.putText(
            annotated_blended,
            cat_name,
            (int(x), int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    orig_preview = COCO_DIR / f"sanity_check_original_{img_info['file_name']}.png"
    overlay_preview = COCO_DIR / f"sanity_check_overlay_{img_info['file_name']}.png"
    annotated_preview = COCO_DIR / f"sanity_check_annotated_{img_info['file_name']}.png"
    try:
        cv2.imwrite(str(orig_preview), orig_img)
        cv2.imwrite(str(overlay_preview), overlay_blended)
        cv2.imwrite(str(annotated_preview), annotated_blended)
    except Exception as e:
        print(f"Warning: could not write preview images: {e}")

    try:
        cv2.imshow("Segmentation Check", annotated_blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print(
            f"OpenCV GUI is unavailable; previews saved to:\n"
            f"  {orig_preview}\n"
            f"  {overlay_preview}\n"
            f"  {annotated_preview}"
        )


if __name__ == "__main__":
    main()
