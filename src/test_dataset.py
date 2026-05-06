import json
import random
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils


ROOT = Path(__file__).resolve().parent.parent
COCO_DIR = ROOT / "datasets" / "coco_format"
ANNOTATION_FILES = [COCO_DIR / "train.json", COCO_DIR / "val.json"]


def resolve_image_path(file_name: str) -> Path:
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
    raise FileNotFoundError(f"No COCO annotations found in: {', '.join(str(p) for p in ANNOTATION_FILES)}")

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


def mask_color(mask_index: int):
    hue = (mask_index * 37) % 180
    color = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def class_color(category_id: int):
    hue = ((category_id - 1) * 37) % 180
    color = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_segmentation(target_img, annotation, color):
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


orig_img = img.copy()
overlay = img.copy()
for idx, ann in enumerate(anns):
    draw_segmentation(overlay, ann, mask_color(idx))

overlay_blended = cv2.addWeighted(overlay, alpha, orig_img, 1 - alpha, 0)

annotated_overlay = img.copy()
for ann in anns:
    draw_segmentation(annotated_overlay, ann, class_color(ann["category_id"]))

annotated_blended = cv2.addWeighted(annotated_overlay, alpha, orig_img, 1 - alpha, 0)

for ann in anns:
    x, y, w, h = ann["bbox"]
    cat_id = ann["category_id"]
    color = class_color(cat_id)
    cat_name = categories.get(cat_id, f"class_{cat_id}")
    cv2.rectangle(annotated_blended, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
    cv2.putText(annotated_blended, cat_name, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
