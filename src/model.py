import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import ResNet50_Weights
from torchvision.ops import DeformConv2d

class DCNv2Wrapper(nn.Module):
    """Native PyTorch implementation of Deformable Convolution v2 for irregular cells."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.offset_mask_conv = nn.Conv2d(
            in_channels, 3 * kernel_size * kernel_size,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        nn.init.constant_(self.offset_mask_conv.weight, 0.)
        nn.init.constant_(self.offset_mask_conv.bias, 0.)

        self.dcn = DeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False
        )

    def forward(self, x):
        out = self.offset_mask_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return self.dcn(x, offset, mask)

def build_dcnv2_mask_rcnn(num_classes=5, min_size=None, max_size=None, anchor_sizes=None):
    """
    Builds the Mask R-CNN with a ResNet50 backbone modified with Deformable Convolutions.
    
    Args:
        num_classes: Number of classes (including background).
        min_size: Minimum size for image resizing. If None, uses torchvision default or provided value.
        max_size: Maximum size for image resizing.
        anchor_sizes: Tuple of anchor sizes for each level of the FPN. 
                      Defaults to ((16,), (32,), (64,), (128,), (256,)).
    """
    if anchor_sizes is None:
        # Optimized for small cells
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))

    # Torchvision defaults if not provided, but we allow None to prevent forced resizing
    # Mask R-CNN typically uses 800/1333, but for tiles we often want to keep them as is.
    # We will pass these to the model; if None, we'll let the model handle its internal defaults
    # or better yet, set them to very small/large values to 'disable' resizing.
    actual_min_size = min_size if min_size is not None else 32
    actual_max_size = max_size if max_size is not None else 4096

    from torchvision.models.detection.anchor_utils import AnchorGenerator
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)
    )

    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        num_classes=num_classes,
        min_size=actual_min_size,
        max_size=actual_max_size,
    )

    # Replace the default anchor generator with our custom one.
    # We do this after initialization because the V2 builder doesn't allow overriding it via kwargs.
    # This works as long as the number of anchors per location (aspect ratios * sizes per level) 
    # remains consistent with what the RPN head was initialized with.
    model.rpn.anchor_generator = anchor_generator

    # Inject DCNv2 into the later stages of the ResNet backbone (Stage 3 and 4)
    # These stages handle higher-level semantic features where deformation is most useful.
    for layer in [model.backbone.body.layer3, model.backbone.body.layer4]:
        for block in layer:
            if hasattr(block, 'conv2'):
                in_c, out_c = block.conv2.in_channels, block.conv2.out_channels
                stride = block.conv2.stride[0]
                block.conv2 = DCNv2Wrapper(in_c, out_c, kernel_size=3, stride=stride, padding=1)

    return model