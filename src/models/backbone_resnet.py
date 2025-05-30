####################
# Copied from C3PO #
####################

import torch
import torch.nn as nn
from torchvision.models import resnet


class Backbone(nn.Module):
    def __init__(self, layer_list):
        super(Backbone, self).__init__()
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.layers[0](x)
        f1 = self.layers[1](f0)
        f2 = self.layers[2](f1)
        f3 = self.layers[3](f2)
        f4 = self.layers[4](f3)
        return (f0, f1, f2, f3, f4)


class ResNet(Backbone):
    def __init__(self, name):
        assert name in ["resnet18", "resnet50"]
        self.name = name
        super(ResNet, self).__init__(get_layers(name))


def get_layers(name):
    if "resnet18" == name:
        replace_stride_with_dilation = [False, False, False]
        model = resnet.__dict__[name](
            pretrained=True,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )
    elif "resnet50" == name:
        replace_stride_with_dilation = [False, True, True]
        model = resnet.__dict__[name](
            pretrained=True,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )
    else:
        raise ValueError(name)

    layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    layer1 = model.layer1
    layer2 = model.layer2
    layer3 = model.layer3
    layer4 = model.layer4
    return [layer0, layer1, layer2, layer3, layer4]
