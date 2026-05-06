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
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

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
FOCAL_ALPHA = 0.25 # Optimized balancing factor
FOCAL_GAMMA = 2.0  # Focusing parameter
# Class weights to handle imbalance (BG, Class 1, 2, 3, 4)
CLASS_WEIGHTS = [1.0, 0.8, 0.8, 2.5, 2.5] 

BATCH_SIZE = 4 
NUM_WORKERS = 4

NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LR_STEP_SIZE = 25
LR_GAMMA = 0.1

MASK_THRESHOLD = 0.5
IOU_THRESHOLDS = [0.5]

# ---------------------------------------------------------
# 1. Custom Focal Loss Injection
# ---------------------------------------------------------
def apply_focal_loss_patch():
    # Store original locally to avoid infinite recursion if re-called
    if not hasattr(roi_heads, "_original_fastrcnn_loss"):
        roi_heads._original_fastrcnn_loss = roi_heads.fastrcnn_loss

    def focal_loss_fastrcnn(class_logits, box_regression, labels, regression_targets):
        labels_cat = torch.cat(labels, dim=0)
        device = class_logits.device
        weights = torch.tensor(CLASS_WEIGHTS, device=device)
        
        ce_loss = F.cross_entropy(class_logits, labels_cat, reduction="none", weight=weights)
        pt = torch.exp(-ce_loss)
        focal_loss = FOCAL_ALPHA * ((1 - pt) ** FOCAL_GAMMA) * ce_loss

        _, box_loss = roi_heads._original_fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        return focal_loss.mean(), box_loss

    roi_heads.fastrcnn_loss = focal_loss_fastrcnn

# ---------------------------------------------------------
# 2. Dataset Setup
# ---------------------------------------------------------
class CellDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        path = self.coco.loadImgs(img_id)[0]['file_name']

        img_full_path = os.path.join(self.root, path)
        img = cv2.imread(img_full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_full_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, masks, labels = [], [], []
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            # Safeguard against degenerate boxes
            if w <= 0 or h <= 0: continue
            
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            masks.append(self.coco.annToMask(ann))
            labels.append(ann['category_id'])

        if self.transforms is not None:
            transformed = self.transforms(image=img, bboxes=boxes, masks=masks, category_ids=labels)
            img = transformed['image']
            boxes = transformed['bboxes']
            masks = transformed['masks']
            labels = transformed['category_ids']

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
    if train:
        return A.Compose([
            # 1. Geometric - Albumentations 2.0 style
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            
            # Robust geometric transforms using A.OneOf
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.OpticalDistortion(distort_limit=0.05, p=0.5),
            ], p=0.3),
            
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
            
            # 2. Lighting and Texture
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.01, p=0.3),
            
            # 3. Microscope Noise/Blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.5),
            ], p=0.2),

            # 4. Format
            A.ToFloat(max_value=255.0),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_area=1))
    else:
        return A.Compose([
            A.ToFloat(max_value=255.0),
            ToTensorV2()
        ])

def collate_fn(batch):
    return tuple(zip(*batch))

# ---------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------
def train_and_evaluate(rank, world_size, args):
    is_distributed = world_size > 1
    if is_distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        print(f"[Rank {rank}] DDP initialized.")
    
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)

    # Patch torchvision locally in this process
    apply_focal_loss_patch()

    run_name = args.run_name if args.run_name else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_ckpt_dir = os.path.join(CHECKPOINT_DIR, run_name)
    if rank == 0:
        os.makedirs(run_ckpt_dir, exist_ok=True)

    train_dataset = CellDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, get_transform(train=True))
    val_dataset = CellDataset(VAL_IMG_DIR, VAL_ANN_FILE, get_transform(train=False))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if is_distributed else None
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=(train_sampler is None),
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        sampler=train_sampler, persistent_workers=(NUM_WORKERS > 0)
    )
    
    if rank == 0:
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
        )

    # Consistent with tiling_utils and preprocessing
    model = build_dcnv2_mask_rcnn(min_size=1, max_size=3000)
    model.to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[rank])

    # Scale LR by world_size for stability in DDP
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * world_size, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    
    # Modern AMP API (Torch 2.x+)
    scaler = torch.amp.GradScaler('cuda')

    metric = MeanAveragePrecision(iou_type="segm", iou_thresholds=IOU_THRESHOLDS, class_metrics=True).to(device)

    start_epoch, best_ap50 = 0, 0.0

    if args.resume and os.path.exists(args.resume):
        if rank == 0: print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint['model_state_dict']
        if is_distributed:
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_ap50 = checkpoint.get('ap50', 0.0)

    if rank == 0:
        wandb.init(project="mask-rcnn-cell-detection", name=run_name, config=vars(args))

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            if is_distributed:
                train_sampler.set_epoch(epoch)
            
            model.train()
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Rank {rank}", disable=(rank != 0))
            
            for images, targets in pbar:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                # Modern AMP API
                with torch.amp.autocast('cuda'):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_losses.append(losses.item())
                if rank == 0:
                    pbar.set_postfix(loss=f"{losses.item():.4f}")

            lr_scheduler.step()

            if rank == 0:
                model.eval()
                metric.reset()
                with torch.no_grad():
                    for images, targets in tqdm(val_loader, desc="Eval", leave=False):
                        images = [img.to(device) for img in images]
                        outputs = model(images)
                        preds = [{"masks": out["masks"].squeeze(1) > MASK_THRESHOLD, "scores": out["scores"], "labels": out["labels"]} for out in outputs]
                        target_metrics = [{"masks": t["masks"].to(device), "labels": t["labels"].to(device)} for t in targets]
                        metric.update(preds, target_metrics)

                ap_metrics = metric.compute()
                current_ap50 = ap_metrics['map_50'].item()
                avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | AP50: {current_ap50:.4f}")
                
                wandb.log({"Train/Loss": avg_loss, "Val/AP50": current_ap50, "LR": optimizer.param_groups[0]['lr']}, step=epoch+1)

                model_to_save = model.module if is_distributed else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'ap50': current_ap50
                }, os.path.join(run_ckpt_dir, "latest_model.pth"))

                if current_ap50 > best_ap50:
                    best_ap50 = current_ap50
                    best_model_path = os.path.join(run_ckpt_dir, "best_model.pth")
                    torch.save(model_to_save.state_dict(), best_model_path)
                    print(f">>> New Best Model Saved to {best_model_path} <<<")
    
    except Exception as e:
        print(f"Error on Rank {rank}: {e}")
        raise e
    finally:
        if rank == 0:
            best_model_path = os.path.join(run_ckpt_dir, "best_model.pth")
            if os.path.exists(best_model_path):
                print(f"Uploading best model to W&B: {best_model_path}")
                wandb.save(best_model_path, base_path=run_ckpt_dir)
            wandb.finish()
        if is_distributed:
            dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Launching DDP with {world_size} GPUs")
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        mp.spawn(train_and_evaluate, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("Starting single-GPU training")
        train_and_evaluate(0, 1, args)

if __name__ == "__main__":
    main()
