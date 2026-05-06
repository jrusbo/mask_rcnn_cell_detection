import os
import json
import torch
import argparse
import warnings
import numpy as np
import skimage.io as sio
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm

from model import build_dcnv2_mask_rcnn

# =========================================================
# GLOBAL INFERENCE CONFIGURATION
# =========================================================
TEST_DIR = "datasets/test_release"
MAP_FILE = os.path.join("datasets", "test_image_name_to_ids.json")
OUTPUT_FILE = "test-results.json"

SCORE_THRESHOLD = 0.05 # Bounding Box Confidence
MASK_THRESHOLD = 0.5 # Pixel Probability Threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Run Instance Segmentation Inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained .pth checkpoint (e.g., checkpoints/2026.../best_ap50_model.pth)"
    )
    return parser.parse_args()


def generate_submission(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.model} on {device}...")

    # Set min_size=1 and max_size=3000 to prevent resizing/upscaling
    # This allows processing full images at their native resolution.
    model = build_dcnv2_mask_rcnn(min_size=1, max_size=3000)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Checkpoint not found at: {args.model}")

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        model_state = checkpoint

    # Handle DDP saved state dict if necessary
    new_state_dict = {}
    for k, v in model_state.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()

    with open(MAP_FILE, 'r') as f:
        image_mappings = json.load(f)

    submission_data = []

    print(f"Running inference on {len(image_mappings)} images...")
    with torch.no_grad():
        for img_info in tqdm(image_mappings, desc="Generating Submission"):
            img_filename = img_info["file_name"]
            img_id = img_info["id"]

            img_path = os.path.join(TEST_DIR, img_filename)
            if not os.path.exists(img_path):
                print(f"Warning: Missing image {img_filename}")
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img = sio.imread(img_path)

            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            
            # Handle grayscale
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)

            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            img_tensor = img_tensor.to(device)

            outputs = model([img_tensor])[0]

            boxes = outputs['boxes'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()
            labels = outputs['labels'].cpu().numpy()
            masks = outputs['masks'].squeeze(1).cpu().numpy()

            for i in range(len(scores)):
                score = float(scores[i])

                if score < SCORE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = boxes[i]
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                binary_mask = (masks[i] > MASK_THRESHOLD).astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(binary_mask))
                rle['counts'] = rle['counts'].decode('utf-8')

                submission_data.append({
                    "image_id": int(img_id),
                    "category_id": int(labels[i]),
                    "bbox": bbox,
                    "score": score,
                    "segmentation": rle
                })

    print(f"\nSaving {len(submission_data)} predictions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(submission_data, f, indent=4)

    print("Inference Complete! Output strictly matches evaluation format.")


if __name__ == "__main__":
    args = parse_args()
    generate_submission(args)