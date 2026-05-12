"""Evaluate a trained checkpoint on the validation set and report AP50 metrics."""

import json

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.auto import tqdm

from model import build_dcnv2_mask_rcnn
from train import CellDataset, get_transform, collate_fn


VAL_IMG_DIR = "datasets/coco_format/images/val"
VAL_ANN_FILE = "datasets/coco_format/val.json"

MODEL_WEIGHTS = r"C:\Users\JoaquinRusBono\Downloads\best_model.pth"

MASK_THRESHOLD = 0.5
MIN_SIZE = 512
MAX_SIZE = 4096
BATCH_SIZE = 4


def evaluate() -> None:
    """Load the validation set, run inference, and print AP50 results.

    Orchestrates a full evaluation workflow:
    1. Loads a trained model from MODEL_WEIGHTS path
    2. Runs inference on the validation dataset
    3. Computes AP50 (Average Precision at IoU=0.5) using torchmetrics
    4. Prints per-class AP50 scores with visual indicators

    Uses configuration constants defined at the module level:
    - MODEL_WEIGHTS: path to the checkpoint file
    - VAL_IMG_DIR, VAL_ANN_FILE: paths to validation data
    - MASK_THRESHOLD, MIN_SIZE, MAX_SIZE: inference parameters
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(VAL_ANN_FILE, "r", encoding="utf-8") as f:
        coco_data = json.load(f)
        class_map = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

    print("Loading Validation Dataset...")
    val_dataset = CellDataset(VAL_IMG_DIR, VAL_ANN_FILE, get_transform(train=False))
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    print(f"Loading Model Weights from {MODEL_WEIGHTS}...")
    model = build_dcnv2_mask_rcnn(min_size=MIN_SIZE, max_size=MAX_SIZE)

    checkpoint = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)

    model.to(device)
    model.eval()

    # By strictly passing iou_thresholds=[0.5], 'map' and 'map_per_class' represent AP50
    metric = MeanAveragePrecision(
        iou_type="segm", iou_thresholds=[0.5], class_metrics=True
    )
    metric.to(device)

    print("Running evaluation (this may take a few minutes)...")
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            preds = [
                {
                    "masks": out["masks"].squeeze(1) > MASK_THRESHOLD,
                    "scores": out["scores"],
                    "labels": out["labels"],
                }
                for out in outputs
            ]

            target_metrics = [
                {"masks": t["masks"], "labels": t["labels"]} for t in targets
            ]

            metric.update(preds, target_metrics)

    print("Computing metrics...")
    result = metric.compute()

    print("\n" + "=" * 50)
    print("              AP50 PER CLASS RESULTS")
    print("=" * 50)

    print(f"Overall AP50 (mAP@0.50) : {result['map'].item():.4f}\n")

    class_ids = result["classes"].tolist()
    class_aps = result["map_per_class"].tolist()

    for cls_id, cls_ap in zip(class_ids, class_aps):
        cls_name = class_map.get(cls_id, f"Class {cls_id}")

        perf_indicator = "🟢" if cls_ap > 0.8 else "🟡" if cls_ap > 0.5 else "🔴"

        print(f"  {perf_indicator} {cls_name:<15} : {cls_ap:.4f}")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    evaluate()
