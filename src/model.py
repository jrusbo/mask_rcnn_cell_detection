"""Model definitions for the cell instance segmentation pipeline."""

from typing import Sequence

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import DeformConv2d


class DCNv2Wrapper(nn.Module):
    """Wrap `DeformConv2d` with a learned offset/mask prediction branch.

    This module learns spatial offsets and modulation masks for deformable convolutions,
    allowing the model to adapt receptive field shapes to irregular cell boundaries.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        """Initialize the deformable convolution wrapper.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel. Defaults to 3.
            stride: Stride of the convolution. Defaults to 1.
            padding: Padding applied to the input. Defaults to 1.
        """

        super().__init__()
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        nn.init.constant_(self.offset_mask_conv.weight, 0.0)
        nn.init.constant_(self.offset_mask_conv.bias, 0.0)

        self.dcn = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply deformable convolution using offsets and a modulation mask.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width).

        Returns:
            Output tensor of shape (batch, out_channels, height', width') where
            height' and width' depend on stride and padding.
        """

        out = self.offset_mask_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return self.dcn(x, offset, mask)


def build_dcnv2_mask_rcnn(
    num_classes: int = 5,
    min_size: int | None = None,
    max_size: int | None = None,
    anchor_sizes: Sequence[Sequence[int]] | None = None,
) -> torch.nn.Module:
    """Build a Mask R-CNN model with a ResNet-50-FPN backbone and DCNv2 blocks.

    This function creates a Mask R-CNN detector pre-trained on ImageNet, replaces
    the default anchor generator with custom small-object-optimized anchors, and
    injects deformable convolution layers into the deeper ResNet stages (3 and 4).

    Args:
        num_classes: Total number of classes including background. Defaults to 5.
        min_size: Minimum size to which images are resized during inference.
            If None, defaults to 800. Defaults to None.
        max_size: Maximum size to which images are resized during inference.
            If None, defaults to 1333. Defaults to None.
        anchor_sizes: Sequence of anchor sizes for each FPN level.
            If None, uses small-object-optimized defaults:
            ((16,), (32,), (64,), (128,), (256,)). Defaults to None.

    Returns:
        A Mask R-CNN model with DCNv2 layers injected into layer3 and layer4
        of the ResNet backbone.
    """

    actual_min_size = min_size if min_size is not None else 800
    actual_max_size = max_size if max_size is not None else 1333

    if anchor_sizes is None:
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes, aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)
    )

    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
        num_classes=num_classes,
        min_size=actual_min_size,
        max_size=actual_max_size,
        box_score_thresh=0.05,  # Keep low to catch faint cells
        box_nms_thresh = 0.55,  # Increased from default 0.5 to allow more overlap
        box_detections_per_img = 300  # Increased from 100 to handle dense clusters
    )

    # Replace the default anchor generator with our custom one.
    # We do this after initialization because the V2 builder doesn't allow overriding it via kwargs.
    # This works as long as the number of anchors per location (aspect ratios * sizes per level)
    # remains consistent with what the RPN head was initialized with.
    model.rpn.anchor_generator = anchor_generator

    # Inject DCNv2 into the later stages of the ResNet backbone (Stage 3 and 4).
    # These stages handle higher-level semantic features where deformation is most useful.
    for layer in [model.backbone.body.layer3, model.backbone.body.layer4]:
        for block in layer:
            if hasattr(block, "conv2"):
                in_c, out_c = block.conv2.in_channels, block.conv2.out_channels
                stride = block.conv2.stride[0]
                block.conv2 = DCNv2Wrapper(
                    in_c, out_c, kernel_size=3, stride=stride, padding=1
                )

    return model
