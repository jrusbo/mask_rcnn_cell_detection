"""Train a Mask R-CNN model for cell instance segmentation."""

import argparse
import os
from datetime import datetime
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.models.detection.roi_heads as roi_heads
import wandb
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.auto import tqdm

from model import build_dcnv2_mask_rcnn


TRAIN_IMG_DIR = "datasets/coco_format/images/train"
TRAIN_ANN_FILE = "datasets/coco_format/train.json"
VAL_IMG_DIR = "datasets/coco_format/images/val"
VAL_ANN_FILE = "datasets/coco_format/val.json"
CHECKPOINT_DIR = "checkpoints"

NUM_CLASSES = 5  # 4 cells + 1 background
FOCAL_ALPHA = 0.25  # Optimized balancing factor
FOCAL_GAMMA = 1.5  # Focusing parameter
# Class weights to handle imbalance (BG, Class 1, 2, 3, 4)
CLASS_WEIGHTS = [1.0, 1.0, 2.5, 2.5, 3.0]

BATCH_SIZE = 8
NUM_WORKERS = 2

NUM_EPOCHS = 45
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LR_STEP_SIZE = 25
LR_GAMMA = 0.1

MASK_THRESHOLD = 0.5
IOU_THRESHOLDS = [0.5]

# Target size matches our padded tile size exactly to ensure scale factor = 1.0 (No resizing)
MIN_SIZE = 512
MAX_SIZE = 4096


def apply_focal_loss_patch() -> None:
    """Patch torchvision's Faster R-CNN classification loss with focal loss.

    Replaces the default cross-entropy loss in the RPN head's classification branch
    with focal loss, which downweights easy examples and focuses on hard negatives.
    This is useful for handling class imbalance in cell detection.
    """

    if not hasattr(roi_heads, "_original_fastrcnn_loss"):
        roi_heads._original_fastrcnn_loss = roi_heads.fastrcnn_loss

    def focal_loss_fastrcnn(
        class_logits: torch.Tensor,
        box_regression: torch.Tensor,
        labels: list[torch.Tensor],
        regression_targets: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute focal classification loss while preserving the box loss.

        Args:
            class_logits: Classification logits of shape (batch_size, num_classes).
            box_regression: Box regression predictions of shape (batch_size, 4 * num_classes).
            labels: List of label tensors, one per image in the batch.
            regression_targets: List of box regression targets, one per image.

        Returns:
            Tuple of (focal_loss, box_loss) where both are scalar tensors.
        """
        labels_cat = torch.cat(labels, dim=0)
        device = class_logits.device
        weights = torch.tensor(CLASS_WEIGHTS, device=device)

        ce_loss = F.cross_entropy(
            class_logits, labels_cat, reduction="none", weight=weights
        )
        pt = torch.exp(-ce_loss)
        focal_loss = FOCAL_ALPHA * ((1 - pt) ** FOCAL_GAMMA) * ce_loss

        _, box_loss = roi_heads._original_fastrcnn_loss(
            class_logits, box_regression, labels, regression_targets
        )
        return focal_loss.mean(), box_loss

    roi_heads.fastrcnn_loss = focal_loss_fastrcnn


class CellDataset(Dataset):
    """Dataset wrapper around COCO annotations and the generated image tiles.

    Loads images and their corresponding instance masks and bounding boxes from
    a COCO-format dataset. Applies optional augmentations during training via
    the Albumentations library.
    """

    def __init__(
        self,
        root: str,
        annotation_file: str,
        transforms: Any | None = None,
    ) -> None:
        """Initialize the COCO dataset wrapper.

        Args:
            root: Directory containing the image files referenced in annotation_file.
            annotation_file: Path to the COCO annotation JSON file.
            transforms: Optional Albumentations Compose object for augmentation.
                Defaults to None.
        """
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index: int) -> tuple[Any, dict[str, torch.Tensor]]:
        """Load and return a single sample from the dataset.

        Args:
            index: Sample index in the dataset (0-based).

        Returns:
            Tuple of (image, target) where:
            - image: Tensor of shape (3, height, width) with pixel values in [0, 1]
            - target: Dictionary with keys:
                - boxes: Tensor of shape (num_instances, 4) in [x1, y1, x2, y2] format
                - masks: Tensor of shape (num_instances, height, width) with uint8 values
                - labels: Tensor of shape (num_instances,) with category IDs
                - image_id: Tensor of shape (1,) with the image ID
        """
        img_id = self.ids[index]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        path = self.coco.loadImgs(img_id)[0]["file_name"]

        img_full_path = os.path.join(self.root, path)
        img = cv2.imread(img_full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_full_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, masks, labels = [], [], []
        for ann in anns:
            xmin, ymin, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            boxes.append([xmin, ymin, xmin + w, ymin + h])
            masks.append(self.coco.annToMask(ann))
            labels.append(ann["category_id"])

        if self.transforms is not None:
            transformed = self.transforms(
                image=img, bboxes=boxes, masks=masks, category_ids=labels
            )
            img = transformed["image"]
            boxes = transformed["bboxes"]
            masks = transformed["masks"]
            labels = transformed["category_ids"]

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
            "image_id": torch.as_tensor([img_id]),
        }
        return img, target

    def __len__(self) -> int:
        """Return the number of COCO images in the dataset.

        Returns:
            Total number of images in the annotation file.
        """

        return len(self.ids)


def get_transform(train: bool) -> A.Compose:
    """Build the Albumentations transform pipeline for training or validation.

    For training, applies a comprehensive set of augmentations including geometric
    transformations (flips, rotations), elastic deformations, and photometric changes.
    For validation, only converts to tensor format without augmentation.

    Args:
        train: If True, returns the training augmentation pipeline. If False,
            returns the minimal validation pipeline (normalization and tensorization).

    Returns:
        An Albumentations Compose object ready to process images and annotations.
    """
    if train:
        return A.Compose(
            [
                # 1. Geometric
                # Cells are orientation-invariant, so flips and 90-deg rotations are safe.
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                # 2. Local Deformations
                # ElasticTransform is excellent for biological cells as it simulates
                # natural shape variation without changing semantic content.
                A.OneOf(
                    [
                        A.ElasticTransform(alpha=1, sigma=50, p=0.5),
                        A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.5),
                        A.OpticalDistortion(distort_limit=0.05, p=0.5),
                    ],
                    p=0.15,
                ),
                # 3. Global Geometric (Affine)
                # Conservative affine: slight shifting and scaling are fine.
                # Scaling (0.9 to 1.1) preserves cell size context while adding robustness.
                # Rotating (-15 to 15) adds diversity beyond 90-deg snaps.
                # Shear is omitted as it can unrealistically stretch cell membranes.
                A.Affine(
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    shear=0,
                    p=0.4,
                ),
                # 4. Intensity / Lighting
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=0.5
                ),
                A.RandomGamma(gamma_limit=(90, 110), p=0.5),
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01, p=0.3
                ),
                # 5. Noise (Simulating sensor noise or focus issues)
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                        A.GaussNoise(std_range=(0.01, 0.03), p=0.5),
                    ],
                    p=0.2,
                ),
                # 6. Format
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["category_ids"], min_area=1
            ),
        )
    else:
        return A.Compose([A.ToFloat(max_value=255.0), ToTensorV2()])


def collate_fn(
    batch: list[tuple[Any, dict[str, torch.Tensor]]],
) -> tuple[tuple[Any, ...], tuple[dict[str, torch.Tensor], ...]]:
    """Batch sampler compatible with variable-length detection targets.

    Converts a list of (image, target) samples into a batch format suitable
    for the Mask R-CNN model, which expects images and targets to remain as
    separate sequences rather than stacked tensors.

    Args:
        batch: List of tuples (image, target) from CellDataset.__getitem__.

    Returns:
        Tuple of (images, targets) where:
        - images is a tuple of image tensors, one per sample
        - targets is a tuple of target dicts, one per sample
    """

    return tuple(zip(*batch))


def train_and_evaluate(rank: int, world_size: int, args: argparse.Namespace) -> None:
    """Train and evaluate the model on one process, optionally under DDP.

    Orchestrates a complete training workflow including:
    1. Multi-GPU DDP initialization if world_size > 1
    2. Model and optimizer setup with focal loss patching
    3. Checkpoint loading if resuming from a prior run
    4. Training loop with mixed precision (AMP)
    5. Validation with AP50 metric computation
    6. Checkpoint saving and W&B logging on rank 0

    Args:
        rank: Process rank (0-based) in the DDP group. Only rank 0 performs
            logging and checkpointing.
        world_size: Total number of processes / GPUs. If > 1, enables DDP.
        args: Parsed command-line arguments containing:
            - resume (str or None): Path to checkpoint to resume from
            - run_name (str or None): Name for this training run (auto-generated if None)
    """

    is_distributed = world_size > 1
    if is_distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=int(rank), world_size=int(world_size))
        print(f"[Rank {rank}] DDP initialized.")

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    apply_focal_loss_patch()

    run_name = (
        args.run_name if args.run_name else datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_ckpt_dir = os.path.join(CHECKPOINT_DIR, run_name)
    if rank == 0:
        os.makedirs(run_ckpt_dir, exist_ok=True)

    train_dataset = CellDataset(
        TRAIN_IMG_DIR, TRAIN_ANN_FILE, get_transform(train=True)
    )
    val_dataset = CellDataset(VAL_IMG_DIR, VAL_ANN_FILE, get_transform(train=False))

    train_sampler = (
        DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        if is_distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=(train_sampler is None),
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_sampler = (
        DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if is_distributed
        else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=val_sampler,
    )

    model = build_dcnv2_mask_rcnn(min_size=MIN_SIZE, max_size=MAX_SIZE)
    model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE * world_size, weight_decay=WEIGHT_DECAY
    )

    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - warmup_epochs, eta_min=1e-6
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
    )

    scaler = torch.amp.GradScaler("cuda")

    metric = MeanAveragePrecision(
        iou_type="segm", iou_thresholds=IOU_THRESHOLDS, class_metrics=True
    ).to(device)

    start_epoch, best_ap50 = 0, 0.0

    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        if is_distributed:
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_ap50 = checkpoint.get("ap50", 0.0)

    if rank == 0:
        wandb.init(project="mask-rcnn-cell-detection", name=run_name, config=vars(args))

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            if is_distributed:
                train_sampler.set_epoch(epoch)

            model.train()
            epoch_losses = []
            pbar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1} Rank {rank}", disable=(rank != 0)
            )

            for images, targets in pbar:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()

                with torch.amp.autocast("cuda"):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_losses.append(losses.item())
                if rank == 0:
                    pbar.set_postfix(loss=f"{losses.item():.4f}")

            lr_scheduler.step()

            model.eval()
            metric.reset()

            val_pbar = (
                tqdm(val_loader, desc="Eval", leave=False) if rank == 0 else val_loader
            )

            with torch.no_grad():
                for images, targets in val_pbar:
                    images = [img.to(device) for img in images]
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
                        {
                            "masks": t["masks"].to(device),
                            "labels": t["labels"].to(device),
                        }
                        for t in targets
                    ]
                    metric.update(preds, target_metrics)

            ap_metrics = metric.compute()
            current_ap50 = ap_metrics["map_50"].item()

            if rank == 0:
                avg_loss = np.mean(epoch_losses)
                print(
                    f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | AP50: {current_ap50:.4f}"
                )

                wandb.log(
                    {
                        "Train/Loss": avg_loss,
                        "Val/AP50": current_ap50,
                        "LR": optimizer.param_groups[0]["lr"],
                    },
                    step=epoch + 1,
                )

                model_to_save = model.module if is_distributed else model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                        "ap50": current_ap50,
                    },
                    os.path.join(run_ckpt_dir, "latest_model.pth"),
                )

                if current_ap50 > best_ap50:
                    best_ap50 = current_ap50
                    best_model_path = os.path.join(run_ckpt_dir, "best_model.pth")
                    torch.save(model_to_save.state_dict(), best_model_path)
                    print(f">>> New Best Model Saved to {best_model_path} <<<")

            if is_distributed:
                dist.barrier(device_ids=[rank])

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


def main() -> None:
    """Parse CLI arguments and launch single-GPU or multi-GPU training.

    Detects the number of available CUDA devices and either starts training
    directly (single GPU) or launches a DDP multiprocessing group (multi-GPU).
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Launching DDP with {world_size} GPUs")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        mp.spawn(
            train_and_evaluate, args=(world_size, args), nprocs=world_size, join=True
        )
    else:
        print("Starting single-GPU training")
        train_and_evaluate(0, 1, args)


if __name__ == "__main__":
    main()
