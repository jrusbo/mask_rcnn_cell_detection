import argparse
import os
import cv2
import torch
import wandb
import numpy as np
import albumentations as A
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models.detection.roi_heads as roi_heads
from datetime import datetime
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

from model import build_dcnv2_mask_rcnn

# =========================================================
# GLOBAL HYPERPARAMETERS & CONFIGURATION
# =========================================================
TRAIN_IMG_DIR = "datasets/coco_format/images/train"
TRAIN_ANN_FILE = "datasets/coco_format/train.json"
VAL_IMG_DIR = "datasets/coco_format/images/val"
VAL_ANN_FILE = "datasets/coco_format/val.json"
CHECKPOINT_DIR = "checkpoints"

NUM_CLASSES = 5  # 4 cells + 1 background
FOCAL_ALPHA = 0.25  # Imbalance balancing factor for minority classes
FOCAL_GAMMA = 2.0  # Focusing parameter for hard examples
MIN_IMAGE_SIZE = 100  # Mask R-CNN internal resize min
MAX_IMAGE_SIZE = 800  # Mask R-CNN internal resize max

BATCH_SIZE = 12
NUM_WORKERS = 6

NUM_EPOCHS = 60
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
LR_STEP_SIZE = 10  # Epochs before dropping Learning Rate
LR_GAMMA = 0.1  # Factor to multiply LR by on step

MASK_THRESHOLD = 0.5  # Binarization threshold for predicted masks
IOU_THRESHOLDS = [0.5]  # Target metric: AP50

# ---------------------------------------------------------
# 1. Custom Focal Loss Injection (Monkey-Patch)
# ---------------------------------------------------------
original_fastrcnn_loss = roi_heads.fastrcnn_loss


def focal_loss_fastrcnn(class_logits, box_regression, labels, regression_targets):
    """Replaces standard Cross Entropy with Focal Loss to combat extreme class imbalance."""
    labels_cat = torch.cat(labels, dim=0)

    ce_loss = F.cross_entropy(class_logits, labels_cat, reduction="none")
    pt = torch.exp(-ce_loss)

    focal_loss = FOCAL_ALPHA * ((1 - pt) ** FOCAL_GAMMA) * ce_loss

    _, box_loss = original_fastrcnn_loss(class_logits, box_regression, labels, regression_targets)

    return focal_loss.mean(), box_loss

roi_heads.fastrcnn_loss = focal_loss_fastrcnn

# ---------------------------------------------------------
# 3. Dataset Setup
# ---------------------------------------------------------
class CellDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None, is_train=True):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.is_train = is_train

    def __getitem__(self, index):
        img_id = self.ids[index]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        path = self.coco.loadImgs(img_id)[0]['file_name']

        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, masks, labels = [], [], []
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            masks.append(self.coco.annToMask(ann))
            labels.append(ann['category_id'])

        if self.transforms is not None:
            if self.is_train:
                transformed = self.transforms(image=img, bboxes=boxes, masks=masks, category_ids=labels)
                img = transformed['image']
                boxes = transformed['bboxes']
                masks = transformed['masks']
                labels = transformed['category_ids']
            else:
                transformed = self.transforms(image=img)
                img = transformed['image']

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, img.shape[1], img.shape[2]), dtype=torch.uint8)
        else:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "masks": masks,
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.as_tensor([img_id])
        }

        return img, target

    def __len__(self):
        return len(self.ids)


def get_transform(train):
    """
    Bulletproof Albumentations pipeline tailored specifically for
    fragmented medical cell microscopy.
    """
    if train:
        return A.Compose([
            # 1. Structural
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            # 2. Biological Morphology
            A.ElasticTransform(p=0.3),
            A.GridDistortion(p=0.3),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-30, 30), p=0.4),

            # 3. Textural & Lighting
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05, hue=0.01, p=0.4),

            # 4. Microscope Artifacts
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(p=0.2),

            # 5. Format for PyTorch Mask R-CNN
            A.ToFloat(max_value=255.0),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    else:
        return A.Compose([
            A.ToFloat(max_value=255.0),
            ToTensorV2()
        ])


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------
# 4. Training & Validation Loop
# ---------------------------------------------------------
def train_and_evaluate(resume_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_ckpt_dir = os.path.join(CHECKPOINT_DIR, run_name)
    os.makedirs(run_ckpt_dir, exist_ok=True)

    use_persistent = NUM_WORKERS > 0
    train_dataset = CellDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, get_transform(train=True), is_train=True)
    val_dataset = CellDataset(VAL_IMG_DIR, VAL_ANN_FILE, get_transform(train=False), is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              collate_fn=collate_fn, pin_memory=True, persistent_workers=use_persistent)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            collate_fn=collate_fn, pin_memory=True, persistent_workers=use_persistent)

    model = build_dcnv2_mask_rcnn()

    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    metric = MeanAveragePrecision(iou_type="segm", iou_thresholds=IOU_THRESHOLDS, class_metrics=True).to(device)

    start_epoch, best_ap50 = 0, 0.0

    if resume_path and os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])  # Restores learning rate exactly
        start_epoch = checkpoint['epoch'] + 1
        best_ap50 = checkpoint.get('ap50', 0.0)

        run_ckpt_dir = os.path.dirname(resume_path)
        run_name = os.path.basename(run_ckpt_dir)
        print(f"Resuming from epoch {start_epoch}")
    else:
        print(f"Starting fresh training run: {run_name}")

    # 3. Initialize W&B with matching run name
    wandb.init(
        project="mask-rcnn-cell-detection",
        name=run_name,
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "focal_alpha": FOCAL_ALPHA,
            "focal_gamma": FOCAL_GAMMA,
            "resumed_from": resume_path
        }
    )

    latest_ckpt = os.path.join(run_ckpt_dir, "latest_model.pth")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # -- TRAIN --
        model.train()
        total_loss = 0

        train_components = {"loss_classifier": 0, "loss_box_reg": 0, "loss_mask": 0, "loss_objectness": 0,
                            "loss_rpn_box_reg": 0}

        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, targets in train_pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            for k, v in loss_dict.items():
                train_components[k] += v.item()

            train_pbar.set_postfix(loss=f"{losses.item():.4f}")

        lr_scheduler.step()

        num_batches = len(train_loader)
        train_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f}")

        wandb_train_logs = {"Train/Total_Loss": train_loss}
        for k, v in train_components.items():
            wandb_train_logs[f"Train_Breakdown/{k}"] = v / num_batches
        wandb.log(wandb_train_logs, step=epoch + 1)

        # -- EVALUATE --
        model.eval()
        metric.reset()
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", leave=False)
            for images, targets in val_pbar:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)

                preds = [{"masks": out["masks"].squeeze(1) > MASK_THRESHOLD, "scores": out["scores"],
                          "labels": out["labels"]} for out in outputs]
                target_metrics = [{"masks": t["masks"], "labels": t["labels"]} for t in targets]
                metric.update(preds, target_metrics)

        ap_metrics = metric.compute()
        current_ap50 = ap_metrics['map_50'].item()
        current_ap75 = ap_metrics['map_75'].item()
        current_recall = ap_metrics['mar_100'].item()

        print(f"Epoch {epoch + 1} | Val AP50: {current_ap50:.4f} | Recall: {current_recall:.4f}")

        log_metrics = {
            "Val/mAP_50_Overall": current_ap50,
            "Val/mAP_75_Strict": current_ap75,
            "Val/Recall_MAR100": current_recall,
            "LR": optimizer.param_groups[0]['lr']
        }

        if 'map_per_class' in ap_metrics:
            for idx, class_ap in enumerate(ap_metrics['map_per_class']):
                log_metrics[f"Val_by_Class/mAP_50_Class_{idx + 1}"] = class_ap.item()

        wandb.log(log_metrics, step=epoch + 1)

        # -- CHECKPOINTING --
        model_to_save = model.module if hasattr(model, 'module') else model

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'ap50': current_ap50
        }, latest_ckpt)

        if current_ap50 > best_ap50:
            best_ap50 = current_ap50
            best_model_path = os.path.join(run_ckpt_dir, "best_ap50_model.pth")
            torch.save(model_to_save.state_dict(), best_model_path)
            print(f">>> New Best Model Saved to {best_model_path} <<<")

            wandb.save(best_model_path, base_path=run_ckpt_dir)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cell Instance Segmentation Mask R-CNN")
    parser.add_argument("--resume", type=str, default=None, help="Path to .pth checkpoint to resume from")
    args = parser.parse_args()

    train_and_evaluate(resume_path=args.resume)