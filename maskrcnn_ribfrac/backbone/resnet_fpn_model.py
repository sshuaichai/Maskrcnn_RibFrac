import os

import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d

from .feature_pyramid_network import BackboneWithFPN, LastLevelMaxPool
import torchvision.models.detection.mask_rcnn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = norm_layer(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = norm_layer(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = norm_layer(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample,
                            stride=stride, norm_layer=norm_layer))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def overwrite_eps(model, eps):
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


def resnet50_fpn_backbone(pretrain_path="",
                          norm_layer=nn.BatchNorm2d,
                          trainable_layers=3,
                          returned_layers=None,
                          extra_blocks=None):
    """
    Build a resnet50_fpn backbone.
    Args:
        pretrain_path: Pretrained weights for resnet50, empty by default if not used.
        norm_layer: Default is nn.BatchNorm2d. If the GPU has limited memory and the batch size cannot be set large,
                    it is recommended to use FrozenBatchNorm2d (default is nn.BatchNorm2d).
                    (https://download.pytorch.org/models/resnet50-0676ba61.pth)
        trainable_layers: Specifies which layers to train.
        returned_layers: Specifies which layers' outputs need to be returned.
        extra_blocks: Additional layers to be added on top of the output feature layers.

    Returns:

    """
    resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3],  # ResNet101 (Bottleneck, [3, 4, 23, 3])
                             include_top=False,
                             norm_layer=norm_layer)

    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(resnet_backbone, 0.0)

    if pretrain_path != "":
        assert os.path.exists(pretrain_path), "{} does not exist.".format(pretrain_path)
        # Load pretrained weights
        print(resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False))

    # Select layers that won't be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # If training all layers, don't forget that there is a bn1 layer after conv1
    if trainable_layers == 5:
        layers_to_train.append("bn1")

    # Freeze layers
    for name, parameter in resnet_backbone.named_parameters():
        # Only train layers not in the layers_to_train list
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    # Ensure the number of returned feature layers is greater than 0 and less than 5
    assert min(returned_layers) > 0 and max(returned_layers) < 5

    # Return layers
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    # in_channel for layer4's output feature map channel = 2048
    in_channels_stage2 = resnet_backbone.in_channel // 8  # 256
    # Record the channels of each feature layer provided to FPN by resnet50
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    # The output channels for each feature layer after FPN
    out_channels = 256
    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)


def resnet101_fpn_backbone(pretrain_path="",
                          norm_layer=nn.BatchNorm2d,
                          trainable_layers=3,
                          returned_layers=None,
                          extra_blocks=None):
    """
    Build a resnet101_fpn backbone.
    Args:
        pretrain_path: Pretrained weights for resnet101, empty by default if not used.
        norm_layer: Default is nn.BatchNorm2d. If the GPU has limited memory and the batch size cannot be set large,
                    it is recommended to use FrozenBatchNorm2d (default is nn.BatchNorm2d).
                    (https://download.pytorch.org/models/resnet101-63fe2227.pth)
        trainable_layers: Specifies which layers to train.
        returned_layers: Specifies which layers' outputs need to be returned.
        extra_blocks: Additional layers to be added on top of the output feature layers.

    Returns:

    """
    resnet_backbone = ResNet(Bottleneck, [3, 4, 23, 3],  # ResNet101 (Bottleneck, [3, 4, 23, 3])
                             include_top=False,
                             norm_layer=norm_layer)

    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(resnet_backbone, 0.0)

    if pretrain_path != "":
        assert os.path.exists(pretrain_path), "{} does not exist.".format(pretrain_path)
        # Load pretrained weights
        print(resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False))

    # Select layers that won't be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # If training all layers, don't forget that there is a bn1 layer after conv1
    if trainable_layers == 5:
        layers_to_train.append("bn1")

    # Freeze layers
    for name, parameter in resnet_backbone.named_parameters():
        # Only train layers not in the layers_to_train list
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    # Ensure the number of returned feature layers is greater than 0 and less than 5
    assert min(returned_layers) > 0 and max(returned_layers) < 5

    # Return layers
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    # in_channel for layer4's output feature map channel = 2048
    in_channels_stage2 = resnet_backbone.in_channel // 8  # 256
    # Record the channels of each feature layer provided to FPN by resnet101
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    # The output channels for each feature layer after FPN
    out_channels = 256
    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)


def resnet152_fpn_backbone(pretrain_path="",
                          norm_layer=nn.BatchNorm2d,
                          trainable_layers=3,
                          returned_layers=None,
                          extra_blocks=None):
    """
    Build a resnet152_fpn backbone.
    Args:
        pretrain_path: Pretrained weights for resnet152, empty by default if not used.
        norm_layer: Default is nn.BatchNorm2d. If the GPU has limited memory and the batch size cannot be set large,
                    it is recommended to use FrozenBatchNorm2d (default is nn.BatchNorm2d).
                    (https://download.pytorch.org/models/resnet152-394f9c45.pth)
        trainable_layers: Specifies which layers to train.
        returned_layers: Specifies which layers' outputs need to be returned.
        extra_blocks: Additional layers to be added on top of the output feature layers.

    Returns:

    """
    resnet_backbone = ResNet(Bottleneck, [3, 8, 36, 3],  # ResNet152 (Bottleneck, [3, 8, 36, 3])
                             include_top=False,
                             norm_layer=norm_layer)

    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(resnet_backbone, 0.0)

    if pretrain_path != "":
        assert os.path.exists(pretrain_path), "{} does not exist.".format(pretrain_path)
        # Load pretrained weights
        print(resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False))

    # Select layers that won't be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # If training all layers, don't forget that there is a bn1 layer after conv1
    if trainable_layers == 5:
        layers_to_train.append("bn1")

    # Freeze layers
    for name, parameter in resnet_backbone.named_parameters():
        # Only train layers not in the layers_to_train list
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    # Ensure the number of returned feature layers is greater than 0 and less than 5
    assert min(returned_layers) > 0 and max(returned_layers) < 5

    # Return layers
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    # in_channel for layer4's output feature map channel = 2048
    in_channels_stage2 = resnet_backbone.in_channel // 8  # 256
    # Record the channels of each feature layer provided to FPN by resnet152
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    # The output channels for each feature layer after FPN
    out_channels = 256
    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
