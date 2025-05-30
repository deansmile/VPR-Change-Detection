"""
Change Detection Model Module:

This module contains models that solve image change detection problem, where
image change detection is find the difference between a image pair.

"""
import sys
import os

# Add the parent directory to the system path
sys.path.append("/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/grounding_dino")
sys.path.append("/scratch/ds5725/alvpr/Robust-Scene-Change-Detection")
from groundingdino.util.inference import load_model

import torch
import torch.nn as nn

from .backbone_dinov2 import TwoDino
from .coattention import TwoCoAttention
from .merge_temporal_feature import MTF
from .TANet import TwoTemporalAttention
from .transformer import (
    TwoCrossAttention,
    CrossAttentionBlock,
)

# sys.path.append("/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/CYWS-3D")
# from modules.registeration_module import FeatureRegisterationModule
# from easydict import EasyDict
# import yaml

# def get_easy_dict_from_yaml_file(path_to_yaml_file):
#     """
#     Reads a yaml and returns it as an easy dict.
#     """
#     with open(path_to_yaml_file, "r") as stream:
#         yaml_file = yaml.safe_load(stream)
#     return EasyDict(yaml_file)

# def check_facet(facet):
#     if facet in {None, "query", "key", "value", "token"}:
#         return
#     raise ValueError(
#         'facet must be one of {None, "query", "key", "value", "token"}'
#     )


class DinoEncoder(nn.Module):

    def __init__(
        self,
        dino1,
        layer1,
        facet1="query",
        dino2=None,
        layer2=None,
        facet2="query",
    ):

        super(DinoEncoder, self).__init__()

        dino2 = dino1 if dino2 is None else dino2
        layer2 = layer1 if layer2 is None else layer2
        self.backbone = TwoDino(dino1, layer1, dino2, layer2)

        # check_facet(facet1)
        # check_facet(facet2)

        self.facet1 = facet1
        self.facet2 = facet2

    def forward(self, img_1, img_2):

        self.backbone(img_1, img_2)

        # (batch, row, col, num_features)
        x1 = self.backbone.get_facet_from_dino1(self.facet1)
        x2 = self.backbone.get_facet_from_dino2(self.facet2)

        # (batch, num_features, row, col)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)

        return x1, x2


class CrossAttention(DinoEncoder):

    def __init__(
        self,
        dino1,
        layer1,
        facet1="query",
        dino2=None,
        layer2=None,
        facet2="query",
        num_heads=1,
        dropout_rate=0.1,
        target_shp=(504, 504),
        num_blocks=1,
        **kwargs,
    ):

        # initialize TwoDinoSingleUnet (backbone)
        super().__init__(
            dino1=dino1,
            layer1=layer1,
            facet1=facet1,
            dino2=dino2,
            layer2=layer2,
            facet2=facet2,
        )

        self.grounding_dino = load_model("/scratch/ds5725/alvpr/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
            "/scratch/ds5725/alvpr/GroundingDINO/weights/groundingdino_swint_ogc.pth")

        for param in self.grounding_dino.parameters():
            param.requires_grad = False

        print("Unfreeze fusion_layers")
        for param in self.grounding_dino.transformer.encoder.fusion_layers.parameters():
            param.requires_grad = True

        # Unfreeze text_layers
        for param in self.grounding_dino.transformer.encoder.text_layers.parameters():
            param.requires_grad = True

        # configs = get_easy_dict_from_yaml_file("/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/CYWS-3D/config.yml")
        # self.feature_register = FeatureRegisterationModule(configs)

        ##### interactor #####
        ca0s = []

        for _ in range(num_blocks):

            ca0s.append(
                TwoCrossAttention(
                    self.backbone.dino_1.num_features,
                    num_heads,
                    dropout_rate,
                    self.backbone.dino_1.num_features,
                    num_heads,
                    dropout_rate,
                )
            )

        self.ca0s = nn.ModuleList(ca0s)

        ##### decoder ######

        self.conv1 = nn.Conv2d(
            self.backbone.dino_1.num_features * 2,
            self.backbone.dino_1.num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            self.backbone.dino_1.num_features, 2, kernel_size=1
        )

        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(
            size=target_shp, mode="bilinear", align_corners=False
        )

    @staticmethod
    def _reshape_before_CA(x):
        batch, feat, m, n = x.shape
        x = x.permute(0, 2, 3, 1)  # (batch, m, n, feat)
        x = x.reshape(batch, m * n, feat)
        return (batch, feat, m, n), x

    @staticmethod
    def _reshape_after_CA(x, origin_shp):
        x = x.permute(0, 2, 1)  # (batch, feat, m * n)
        return x.reshape(*origin_shp)  # (batch, feat, m, n)

    @staticmethod
    def apply_cross_attention(self, x_origin, y_origin, cas):
        shp_x, x = self._reshape_before_CA(x_origin)
        shp_y, y = self._reshape_before_CA(y_origin)

        for ca in cas:
            x, y = ca(x, y, y, y, x, x)

        x = self._reshape_after_CA(x, shp_x)
        y = self._reshape_after_CA(y, shp_y)
        return x, y

    def create_dummy_batch_info(self, feature):
        B, C, H, W = feature.shape
        device = feature.device
        identity_transform = torch.eye(3, device=device)  # 3x3 identity matrix

        return {
            "registration_strategy": ["2d"] * B,
            "points1": [None] * B,
            "points2": [None] * B,
            "intrinsics1": [None] * B,
            "intrinsics2": [None] * B,
            "rotation1": [None] * B,
            "rotation2": [None] * B,
            "position1": [None] * B,
            "position2": [None] * B,
            "depth1": torch.ones((B, 1, H, W), device=device),
            "depth2": torch.ones((B, 1, H, W), device=device),
            "transfm2d_1_to_2": [identity_transform.clone() for _ in range(B)],
            "transfm2d_2_to_1": [identity_transform.clone() for _ in range(B)],
        }

    def forward(self, img_1, img_2, caption):
        # print(f"Input img_1 shape: {img_1.shape}")
        # print(f"Input img_2 shape: {img_2.shape}")
        
        dino_feat_1_0, dino_feat_2_0 = super().forward(img_1, img_2)

        # print(f"DINOv2 feature shape from img_1: {dino_feat_1_0.shape}")
        # print(f"DINOv2 feature shape from img_2: {dino_feat_2_0.shape}")
        # if caption.strip():  # only run if caption is not empty or whitespace
        caption = [c if c.strip() else "." for c in caption]
        dino_feat_1_0 = self.grounding_dino.improve_dino_feature(dino_feat_1_0, caption)
        # print(f"DINOv2 feature shape from img_1 after grounding_dino: {dino_feat_1_0.shape}")

        # batch_info = self.create_dummy_batch_info(dino_feat_1_0)
        # dino_feat_1_0, dino_feat_2_0 = self.feature_register(batch_info, dino_feat_1_0, dino_feat_2_0)

        # cross attention on (row, col)
        x_1_0, x_2_0 = self.apply_cross_attention(
            self, dino_feat_1_0, dino_feat_2_0, self.ca0s
        )

        # (batch, 2f, r, c)
        x = torch.concatenate([x_1_0, x_2_0], dim=1)

        # print(f"Concatenated feature shape: {x.shape}")
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)  # (batch, *target_shp, 2)
        return x


class SingleCrossAttention1(DinoEncoder):

    def __init__(
        self,
        dino1,
        layer1,
        facet1="query",
        dino2=None,
        layer2=None,
        facet2="query",
        num_heads=1,
        dropout_rate=0.1,
        target_shp=(504, 504),
        **kwargs,
    ):

        # initialize TwoDinoSingleUnet (backbone)
        super().__init__(
            dino1=dino1,
            layer1=layer1,
            facet1=facet1,
            dino2=dino2,
            layer2=layer2,
            facet2=facet2,
        )

        ##### interactor #####

        self.ca = CrossAttentionBlock(
            embed_dim=self.backbone.dino_1.num_features,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        ##### decoder ######

        self.conv1 = nn.Conv2d(
            self.backbone.dino_1.num_features,
            self.backbone.dino_1.num_features,
            kernel_size=3,
        )
        self.conv2 = nn.Conv2d(
            self.backbone.dino_1.num_features, 2, kernel_size=1
        )

        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(
            size=target_shp, mode="bilinear", align_corners=False
        )

    @staticmethod
    def _reshape_before_CA(x):
        batch, feat, m, n = x.shape
        x = x.permute(0, 2, 3, 1)  # (batch, m, n, feat)
        x = x.reshape(batch, m * n, feat)
        return (batch, feat, m, n), x

    @staticmethod
    def _reshape_after_CA(x, origin_shp):
        x = x.permute(0, 2, 1)  # (batch, feat, m * n)
        return x.reshape(*origin_shp)  # (batch, feat, m, n)

    def forward(self, img_1, img_2):

        # (batch, num_features, row, col)
        dino_feat_1_0, dino_feat_2_0 = super().forward(img_1, img_2)

        # cross attention on (row, col)
        shp_x, x = self._reshape_before_CA(dino_feat_1_0)
        shp_y, y = self._reshape_before_CA(dino_feat_2_0)

        # (batch, row * col, num_features)
        x = self.ca(x, y, y)

        # (batch, num_feature, row, col)
        x = self._reshape_after_CA(x, shp_x)

        # (batch, 2, row, col)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)  # (batch, *target_shp, 2)
        return x


class MergeTemporal(DinoEncoder):

    def __init__(
        self,
        dino1,
        layer1,
        facet1="query",
        dino2=None,
        layer2=None,
        facet2="query",
        target_shp=(504, 504),
        **kwargs,
    ):

        # initialize TwoDinoSingleUnet (backbone)
        super().__init__(
            dino1=dino1,
            layer1=layer1,
            facet1=facet1,
            dino2=dino2,
            layer2=layer2,
            facet2=facet2,
        )

        ##### interactor #####
        opts = {"mode": "id", "kernel_size": 3}

        self.mtf0 = MTF(self.backbone.dino_1.num_features, **opts)

        ##### decoder ######

        self.conv1 = nn.Conv2d(
            self.backbone.dino_1.num_features * 2,
            self.backbone.dino_1.num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            self.backbone.dino_1.num_features, 2, kernel_size=1
        )

        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(
            size=target_shp, mode="bilinear", align_corners=False
        )

    def forward(self, img_1, img_2):

        dino_feat_1_0, dino_feat_2_0 = super().forward(img_1, img_2)

        # MTF on (row, col)
        x_1_0 = self.mtf0(dino_feat_1_0, dino_feat_2_0)
        x_2_0 = self.mtf0(dino_feat_2_0, dino_feat_1_0)

        # (batch, 2f, r, c)
        x = torch.concatenate([x_1_0, x_2_0], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)  # (batch, *target_shp, 2)
        return x


class CoAttention(DinoEncoder):

    def __init__(
        self,
        dino1,
        layer1,
        facet1="query",
        dino2=None,
        layer2=None,
        facet2="query",
        target_shp=(504, 504),
        num_blocks=1,
        **kwargs,
    ):

        # initialize TwoDinoSingleUnet (backbone)
        super().__init__(
            dino1=dino1,
            layer1=layer1,
            facet1=facet1,
            dino2=dino2,
            layer2=layer2,
            facet2=facet2,
        )

        ##### interactor #####
        co0s = []

        for _ in range(num_blocks):

            co0s.append(
                TwoCoAttention(
                    input_channels_1=self.backbone.dino_1.num_features,
                    input_channels_2=self.backbone.dino_1.num_features,
                )
            )

        self.co0s = nn.ModuleList(co0s)

        ##### decoder ######

        self.conv1 = nn.Conv2d(
            self.backbone.dino_1.num_features * 2,
            self.backbone.dino_1.num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            self.backbone.dino_1.num_features, 2, kernel_size=1
        )

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(
            size=target_shp, mode="bilinear", align_corners=False
        )

    @staticmethod
    def apply_co_attention(x, y, cos):

        for co in cos:
            x, y = co(x, y)

        return x, y

    def forward(self, img_1, img_2):

        dino_feat_1_0, dino_feat_2_0 = super().forward(img_1, img_2)

        # co-attention on (row, col)
        x_1_0, x_2_0 = self.apply_co_attention(
            dino_feat_1_0, dino_feat_2_0, self.co0s
        )

        # (batch, 2f, r, c)
        x = torch.concatenate([x_1_0, x_2_0], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)  # (batch, *target_shp, 2)
        return x


class TemporalAttention(DinoEncoder):

    def __init__(
        self,
        dino1,
        layer1,
        facet1="query",
        dino2=None,
        layer2=None,
        facet2="query",
        target_shp=(504, 504),
        ta_kernel_size=7,
        **kwargs,
    ):

        # initialize TwoDinoSingleUnet (backbone)
        super().__init__(
            dino1=dino1,
            layer1=layer1,
            facet1=facet1,
            dino2=dino2,
            layer2=layer2,
            facet2=facet2,
        )

        ##### interactor #####

        ta = TwoTemporalAttention(
            self.backbone.dino_1.num_features,
            self.backbone.dino_1.num_features,
            kernel_size=ta_kernel_size,
            stride=1,
            padding=ta_kernel_size // 2,
            groups=4,
            refinement=True,
        )

        self.ta = ta

        ##### decoder ######

        self.conv1 = nn.Conv2d(
            self.backbone.dino_1.num_features * 2,
            self.backbone.dino_1.num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            self.backbone.dino_1.num_features, 2, kernel_size=1
        )

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(
            size=target_shp, mode="bilinear", align_corners=False
        )

    def forward(self, img_1, img_2):

        dino_feat_1_0, dino_feat_2_0 = super().forward(img_1, img_2)

        # co-attention on (row, col)
        x_1_0, x_2_0 = self.ta(dino_feat_1_0, dino_feat_2_0)

        # (batch, 2f, r, c)
        x = torch.concatenate([x_1_0, x_2_0], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)  # (batch, *target_shp, 2)
        return x


class ResNet18CrossAttention(nn.Module):

    def __init__(
        self,
        resnet,
        num_heads=1,
        dropout_rate=0.1,
        target_shp=(512, 512),
        target_feature=128,
        **kwargs,
    ):

        super().__init__()

        # pretrain on ImageNet
        self.backbone = resnet
        self.target_feature = target_feature

        ##### interactor #####
        self.ca = TwoCrossAttention(
            target_feature,
            num_heads,
            dropout_rate,
            target_feature,
            num_heads,
            dropout_rate,
        )

        ##### decoder ######

        self.conv1 = nn.Conv2d(
            target_feature * 2,
            target_feature,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(target_feature, 2, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(
            size=target_shp, mode="bilinear", align_corners=False
        )

    @staticmethod
    def _reshape_before_CA(x):
        batch, feat, m, n = x.shape
        x = x.permute(0, 2, 3, 1)  # (batch, m, n, feat)
        x = x.reshape(batch, m * n, feat)
        return (batch, feat, m, n), x

    @staticmethod
    def _reshape_after_CA(x, origin_shp):
        x = x.permute(0, 2, 1)  # (batch, feat, m * n)
        return x.reshape(*origin_shp)  # (batch, feat, m, n)

    @staticmethod
    def apply_cross_attention(self, x_origin, y_origin, ca):
        shp_x, x = self._reshape_before_CA(x_origin)
        shp_y, y = self._reshape_before_CA(y_origin)

        x, y = ca(x, y, y, y, x, x)

        x = self._reshape_after_CA(x, shp_x)
        y = self._reshape_after_CA(y, shp_y)
        return x, y

    def forward(self, img_1, img_2):

        f1s = self.backbone(img_1)
        f2s = self.backbone(img_2)

        f1s = {
            64: f1s[1],
            128: f1s[2],
            256: f1s[3],
            512: f1s[4],
        }

        f2s = {
            64: f2s[1],
            128: f2s[2],
            256: f2s[3],
            512: f2s[4],
        }

        f1 = f1s[self.target_feature]
        f2 = f2s[self.target_feature]
        f1, f2 = self.apply_cross_attention(self, f1, f2, self.ca)

        # (batch, 2f, r, c)
        x = torch.concatenate([f1, f2], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)  # (batch, *target_shp, 2)
        return x
