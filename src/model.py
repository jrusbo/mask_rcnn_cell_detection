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

def build_dcnv2_mask_rcnn(num_classes=5, min_size=100, max_size=3000):
    """
    Builds the Mask R-CNN with a ResNet50 backbone modified with Deformable Convolutions.
    """
    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
    )

    # Inject DCNv2 into the later stages of the ResNet backbone
    for layer in [model.backbone.body.layer3, model.backbone.body.layer4]:
        for block in layer:
            in_c, out_c = block.conv2.in_channels, block.conv2.out_channels
            stride = block.conv2.stride[0]
            block.conv2 = DCNv2Wrapper(in_c, out_c, kernel_size=3, stride=stride, padding=1)

    return model