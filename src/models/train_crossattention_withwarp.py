#!/usr/bin/env python3
"""
Complete Training script for CrossAttention model for VL-CMU-CD change detection.
Updated with torch.hub.load() and VL-CMU-CD dataset integration.
Integrated with DINOv2-GroundingDINO adapter to fix dimension mismatch (768<->384).
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import logging
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import re

import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add the parent directory to the system path
sys.path.append("/scratch/zl4701/rscd/Robust-Scene-Change-Detection/grounding_dino")
sys.path.append("/scratch/zl4701/rscd/Robust-Scene-Change-Detection")
sys.path.append("/scratch/zl4701/rscd/Robust-Scene-Change-Detection/src/models")
sys.path.append("/scratch/zl4701/rscd/Robust-Scene-Change-Detection/CYWS-3D")

# Original imports
from groundingdino.util.inference import load_model
from modules.registeration_module import FeatureRegisterationModule
from easydict import EasyDict
import yaml

# Import DINOv2-GroundingDINO adapter
from dinov2_grounding_adapter import DINOv2ToGroundingAdapter

# You'll need to import these from your model files
from backbone_dinov2 import TwoDino
from coattention import TwoCoAttention
from merge_temporal_feature import MTF
from TANet import TwoTemporalAttention
from transformer import (
    TwoCrossAttention,
    CrossAttentionBlock,
)

# SuperGlue imports
sys.path.append('/scratch/zl4701/CYWS-3D/SuperGluePretrainedNetwork')
from models.matching import Matching

class LossTracker:
    """损失函数跟踪器"""
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
    def update(self, epoch, train_loss, val_loss):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
    def save_losses(self):
        loss_data = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.save_dir / 'loss_history.json', 'w') as f:
            json.dump(loss_data, f, indent=2)
            
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

def save_prediction_images(model, dataloader, device, save_dir, num_samples=5):
   """保存模型预测结果图片 - 重写版本"""
   save_dir = Path(save_dir)
   save_dir.mkdir(exist_ok=True)
   
   model.eval()
   saved_count = 0
   
   print(f"🖼️ Saving {num_samples} prediction images...")
   
   with torch.no_grad():
       for batch_idx, (img1, img2, captions, targets) in enumerate(dataloader):
           if saved_count >= num_samples:
               break
               
           img1, img2, targets = img1.to(device), img2.to(device), targets.to(device)
           
           try:
               # 获取模型输出
               outputs = model(img1, img2, captions)
               print(f"Raw output shape: {outputs.shape}")
               print(f"Raw output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
               
               # 处理输出维度 [B, H, W, 2] -> [B, 2, H, W]
               if outputs.dim() == 4 and outputs.shape[-1] == 2:
                   outputs = outputs.permute(0, 3, 1, 2)  # [B, 2, H, W]
               
               # 应用softmax并获取预测
               outputs_prob = torch.softmax(outputs, dim=1)  # [B, 2, H, W]
               predictions = torch.argmax(outputs_prob, dim=1)  # [B, H, W]
               confidence = torch.max(outputs_prob, dim=1)[0]  # [B, H, W]
               
               print(f"Prediction shape: {predictions.shape}")
               print(f"Prediction unique values: {torch.unique(predictions)}")
               
               # 保存每个样本
               for i in range(min(img1.shape[0], num_samples - saved_count)):
                   # 转换为numpy并处理
                   img1_np = img1[i].cpu().permute(1, 2, 0).numpy()
                   img2_np = img2[i].cpu().permute(1, 2, 0).numpy()
                   pred_np = predictions[i].cpu().numpy()
                   target_np = targets[i].cpu().numpy()
                   conf_np = confidence[i].cpu().numpy()
                   
                   # 反归一化图像
                   mean = np.array([0.485, 0.456, 0.406])
                   std = np.array([0.229, 0.224, 0.225])
                   img1_np = np.clip(img1_np * std + mean, 0, 1)
                   img2_np = np.clip(img2_np * std + mean, 0, 1)
                   
                   # 打印调试信息
                   print(f"Sample {saved_count}:")
                   print(f"  Target range: [{target_np.min()}, {target_np.max()}]")
                   print(f"  Prediction range: [{pred_np.min()}, {pred_np.max()}]")
                   print(f"  Confidence range: [{conf_np.min():.3f}, {conf_np.max():.3f}]")
                   
                   # 创建可视化 - 2x3布局
                   fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                   
                   # 第一行：输入图像
                   axes[0, 0].imshow(img1_np)
                   axes[0, 0].set_title('Image 1 (t0)')
                   axes[0, 0].axis('off')
                   
                   axes[0, 1].imshow(img2_np)
                   axes[0, 1].set_title('Image 2 (t1)')
                   axes[0, 1].axis('off')
                   
                   axes[0, 2].text(0.1, 0.5, f"Caption:\n{captions[i]}", 
                                  transform=axes[0, 2].transAxes, fontsize=10,
                                  verticalalignment='center', wrap=True)
                   axes[0, 2].set_title('Text Description')
                   axes[0, 2].axis('off')
                   
                   im1 = axes[1, 0].imshow(target_np, cmap='RdYlBu', vmin=0, vmax=1)
                   axes[1, 0].set_title(f'Ground Truth\n(Changed pixels: {target_np.sum()}/{target_np.size})')
                   axes[1, 0].axis('off')
                   plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
                   
                   im2 = axes[1, 1].imshow(pred_np, cmap='RdYlBu', vmin=0, vmax=1)
                   axes[1, 1].set_title(f'Prediction\n(Changed pixels: {pred_np.sum()}/{pred_np.size})')
                   axes[1, 1].axis('off')
                   plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
                   
                   im3 = axes[1, 2].imshow(conf_np, cmap='viridis', vmin=0, vmax=1)
                   axes[1, 2].set_title(f'Confidence\n(Avg: {conf_np.mean():.3f})')
                   axes[1, 2].axis('off')
                   plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
                   
                   plt.tight_layout()
                   
                   save_path = save_dir / f'prediction_{saved_count:03d}.png'
                   plt.savefig(save_path, dpi=150, bbox_inches='tight')
                   plt.close()
                   
                   fig_mask, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                   
                   ax1.imshow(target_np, cmap='hot', vmin=0, vmax=1)
                   ax1.set_title('Ground Truth Mask')
                   ax1.axis('off')
                   
                   ax2.imshow(pred_np, cmap='hot', vmin=0, vmax=1)
                   ax2.set_title('Predicted Mask')
                   ax2.axis('off')
                   
                   plt.tight_layout()
                   mask_save_path = save_dir / f'mask_comparison_{saved_count:03d}.png'
                   plt.savefig(mask_save_path, dpi=150, bbox_inches='tight')
                   plt.close()
                   
                   saved_count += 1
                   print(f"  ✓ Saved prediction_{saved_count-1:03d}.png")
                   
           except Exception as e:
               print(f"❌ Error processing batch {batch_idx}: {e}")
               import traceback
               traceback.print_exc()
               continue
               
           if saved_count >= num_samples:
               break
   
   print(f"✅ Successfully saved {saved_count} prediction images to {save_dir}")
   print(f"   📁 Files: prediction_000.png to prediction_{saved_count-1:03d}.png")
   print(f"   📁 Mask comparisons: mask_comparison_000.png to mask_comparison_{saved_count-1:03d}.png")


class SuperGlueWarpEstimator:
    """SuperGlue-based homography estimation for image registration"""
    
    def __init__(self, superglue_config, device='cuda'):
        self.device = device
        self.matching = Matching(superglue_config).eval().to(device)
     
    def read_and_preprocess(self, image_path):
        """Preprocess image for SuperGlue"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        assert image is not None, f"Image not found: {image_path}"
        h, w = image.shape
        scale = min(640 / h, 480 / w)
        resized = cv2.resize(image, (int(scale * w), int(scale * h)))
        padded = np.zeros((640, 480), dtype=np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        tensor = torch.from_numpy(padded / 255.).float()[None, None].to(self.device)
        return tensor, padded
     
    def get_warp_matrix(self, img_path0, img_path1):
        """Get homography matrix from image paths"""
        img0_tensor, _ = self.read_and_preprocess(img_path0)
        img1_tensor, _ = self.read_and_preprocess(img_path1)
        data = {'image0': img0_tensor, 'image1': img1_tensor}
         
        with torch.no_grad():
            pred = self.matching(data)
         
        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
         
        valid = matches > -1
        if valid.sum() < 4:
            return torch.eye(3, device=self.device)
         
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        H, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
        if H is None:
            return torch.eye(3, device=self.device)
        return torch.from_numpy(H).float().to(self.device)
     
    def get_warp_matrix_from_tensor(self, img_tensor0, img_tensor1):
        """Get homography matrix from image tensors"""
        def to_gray_np(tensor):
            img = tensor.detach().cpu().numpy()
            img = np.mean(img, axis=0)
            img = (img * 255).astype(np.uint8)
            return img
         
        img0 = to_gray_np(img_tensor0)
        img1 = to_gray_np(img_tensor1)
        img0_tensor = torch.from_numpy(img0 / 255.).float()[None, None].to(self.device)
        img1_tensor = torch.from_numpy(img1 / 255.).float()[None, None].to(self.device)
         
        with torch.no_grad():
            pred = self.matching({'image0': img0_tensor, 'image1': img1_tensor})
         
        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        valid = matches > -1
         
        if valid.sum() < 4:
            return torch.eye(3, device=self.device)
         
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        H, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
        if H is None:
            return torch.eye(3, device=self.device)
        return torch.from_numpy(H).float().to(self.device)


def get_easy_dict_from_yaml_file(path_to_yaml_file):
    """Reads a yaml and returns it as an easy dict."""
    with open(path_to_yaml_file, "r") as stream:
        yaml_file = yaml.safe_load(stream)
    return EasyDict(yaml_file)


def check_facet(facet):
    """Validate DINOv2 facet parameter"""
    if facet in {None, "query", "key", "value", "token"}:
        return
    raise ValueError('facet must be one of {None, "query", "key", "value", "token"}')


class DinoEncoder(nn.Module):
    """Base DINOv2 encoder class - Updated to handle pre-loaded models"""
    
    def __init__(self, dino1, layer1, facet1="query", dino2=None, layer2=None, facet2="query"):
        super(DinoEncoder, self).__init__()
        
        # Handle both string model names and pre-loaded models
        if isinstance(dino1, str):
            # Legacy support: load model from string
            print(f"Loading DINOv2 model: {dino1}")
            dino1_model = torch.hub.load("facebookresearch/dinov2", dino1, pretrained=True)
        else:
            # New approach: use pre-loaded model
            dino1_model = dino1
            
        if dino2 is None:
            dino2_model = dino1_model
        elif isinstance(dino2, str):
            print(f"Loading second DINOv2 model: {dino2}")
            dino2_model = torch.hub.load("facebookresearch/dinov2", dino2, pretrained=True)
        else:
            dino2_model = dino2
            
        layer2 = layer1 if layer2 is None else layer2
        self.backbone = TwoDino(dino1_model, layer1, dino2_model, layer2)
        
        check_facet(facet1)
        check_facet(facet2)
        
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
    """CrossAttention model with integrated DINOv2-GroundingDINO adapter"""
    
    def __init__(self, dino1, layer1, facet1="query", dino2=None, layer2=None, facet2="query",
                 num_heads=1, dropout_rate=0.1, target_shp=(504, 504), num_blocks=1, **kwargs):
        
        # Initialize TwoDinoSingleUnet (backbone)
        super().__init__(dino1=dino1, layer1=layer1, facet1=facet1, 
                        dino2=dino2, layer2=layer2, facet2=facet2)

        # SuperGlue configuration
        superglue_config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024,
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }

        self.warp_estimator = SuperGlueWarpEstimator(superglue_config, 
                                                   device='cuda' if torch.cuda.is_available() else 'cpu')

        # GroundingDINO adapter instead of direct grounding_dino
        print("🚀 Loading GroundingDINO with DINOv2 adapter...")
        try:
            grounding_model = load_model(
                    "/scratch/zl4701/rscd/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                    "/scratch/zl4701/rscd/GroundingDINO/weights/groundingdino_swint_ogc.pth"
                )
            
            
            # Create adapter instead of using GroundingDINO directly
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.grounding_adapter = DINOv2ToGroundingAdapter(grounding_model, device)
            
            print("✓ GroundingDINO adapter loaded successfully!")
            
        except Exception as e:
            print(f"⚠ Failed to load GroundingDINO adapter: {e}")
            print("  Training will continue without text-visual enhancement")
            self.grounding_adapter = None

        # Configure GroundingDINO gradients (if adapter loaded successfully)
        if self.grounding_adapter is not None:
            # Freeze GroundingDINO main model
            for param in self.grounding_adapter.grounding_dino.parameters():
                param.requires_grad = False

            # Unfreeze specific layers (if they exist)
            try:
                print("Unfreezing fusion_layers...")
                for param in self.grounding_adapter.grounding_dino.transformer.encoder.fusion_layers.parameters():
                    param.requires_grad = True
            except AttributeError:
                print("No fusion_layers found, skipping...")

            try:
                print("Unfreezing text_layers...")
                for param in self.grounding_adapter.grounding_dino.transformer.encoder.text_layers.parameters():
                    param.requires_grad = True
            except AttributeError:
                print("No text_layers found, skipping...")

            # Adapter parameters are trainable
            for param in self.grounding_adapter.feature_adapter.parameters():
                param.requires_grad = True
            for param in self.grounding_adapter.attention_fusion.parameters():
                param.requires_grad = True
            for param in self.grounding_adapter.feature_enhancer.parameters():
                param.requires_grad = True
            for param in self.grounding_adapter.simple_fusion.parameters():
                param.requires_grad = True

        # Feature registration module
        configs = get_easy_dict_from_yaml_file("/scratch/zl4701/rscd/Robust-Scene-Change-Detection/CYWS-3D/config.yml")
        self.feature_register = FeatureRegisterationModule(configs)

        # Cross-attention blocks
        ca0s = []
        for _ in range(num_blocks):
            ca0s.append(
                TwoCrossAttention(
                    self.backbone.dino_1.num_features, num_heads, dropout_rate,
                    self.backbone.dino_1.num_features, num_heads, dropout_rate,
                )
            )
        self.ca0s = nn.ModuleList(ca0s)

        # Decoder
        self.conv1 = nn.Conv2d(
            self.backbone.dino_1.num_features * 2,
            self.backbone.dino_1.num_features,
            kernel_size=3, stride=1, padding=1,
        )
        self.conv2 = nn.Conv2d(self.backbone.dino_1.num_features, 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(size=target_shp, mode="bilinear", align_corners=False)

    def enhance_features_with_text(self, features, captions):
        """
        Enhance features using DINOv2-GroundingDINO adapter
        Handles dimension conversion: DINOv2(768) <-> GroundingDINO(384)
        
        Args:
            features: DINOv2 features [B, 768, H, W]
            captions: List of text descriptions
            
        Returns:
            enhanced_features: Enhanced features [B, 768, H, W]
        """
        if self.grounding_adapter is None:
            print("⚠ GroundingDINO adapter not available, returning original features")
            return features
        
        # Validate input
        if not captions or all(not c.strip() for c in captions):
            print("⚠ Empty captions, skipping text enhancement")
            return features
        
        print(f"🔄 Enhancing features with text: {captions}")
        print(f"   Input features shape: {features.shape}")
        
        try:
            # Use adapter to enhance features (automatically handles 768->384->768 conversion)
            enhanced_features = self.grounding_adapter.improve_dino_feature(features, captions)
            print(f"✓ Text enhancement successful! Output shape: {enhanced_features.shape}")
            return enhanced_features
            
        except Exception as e:
            print(f"⚠ Text enhancement failed: {e}")
            print(f"  Error type: {type(e).__name__}")
            print("  Returning original features")
            return features

    @staticmethod
    def _reshape_before_CA(x):
        """Reshape features before cross-attention"""
        batch, feat, m, n = x.shape
        x = x.permute(0, 2, 3, 1)  # (batch, m, n, feat)
        x = x.reshape(batch, m * n, feat)
        return (batch, feat, m, n), x

    @staticmethod
    def _reshape_after_CA(x, origin_shp):
        """Reshape features after cross-attention"""
        x = x.permute(0, 2, 1)  # (batch, feat, m * n)
        return x.reshape(*origin_shp)  # (batch, feat, m, n)

    @staticmethod
    def apply_cross_attention(self, x_origin, y_origin, cas):
        """Apply cross-attention between two feature maps"""
        shp_x, x = self._reshape_before_CA(x_origin)
        shp_y, y = self._reshape_before_CA(y_origin)

        for ca in cas:
            x, y = ca(x, y, y, y, x, x)

        x = self._reshape_after_CA(x, shp_x)
        y = self._reshape_after_CA(y, shp_y)
        return x, y

    def create_dummy_batch_info(self, feature):
        """Create dummy batch info for feature registration"""
        B, C, H, W = feature.shape
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
            "depth1": torch.ones((B, 1, H, W), device=feature.device),
            "depth2": torch.ones((B, 1, H, W), device=feature.device),
        }

    def forward(self, img_1, img_2, caption):
        """Forward pass of the CrossAttention model"""
        print(f"Input img_1 shape: {img_1.shape}")
        print(f"Input img_2 shape: {img_2.shape}")
        
        # Extract DINOv2 features
        dino_feat_1_0, dino_feat_2_0 = super().forward(img_1, img_2)
        print(f"DINOv2 feature shape from img_1: {dino_feat_1_0.shape}")
        print(f"DINOv2 feature shape from img_2: {dino_feat_2_0.shape}")
        
        # Use adapter for text-guided feature enhancement (fixes 768->384->768 dimension issue)
        if any(c.strip() for c in caption):
            print("🚀 Applying text-guided feature enhancement...")
            dino_feat_1_0 = self.enhance_features_with_text(dino_feat_1_0, caption)
            # Optional: also enhance second image features
            # dino_feat_2_0 = self.enhance_features_with_text(dino_feat_2_0, caption)
        else:
            print("⚠ All captions are empty, skipping text enhancement")

        # SuperGlue registration
        print("🔄 Computing warp matrices with SuperGlue...")
        warp_matrices_1_to_2 = []
        warp_matrices_2_to_1 = []
        B = img_1.shape[0]

        for i in range(B):
            H12 = self.warp_estimator.get_warp_matrix_from_tensor(img_1[i], img_2[i])
            H21 = torch.linalg.pinv(H12)
            warp_matrices_1_to_2.append(H12)
            warp_matrices_2_to_1.append(H21)

        batch_info = {
                "transfm2d_1_to_2": warp_matrices_1_to_2,
                "transfm2d_2_to_1": warp_matrices_2_to_1,
                "shape": list(dino_feat_1_0.shape[-2:]),  # Convert to list
                "registration_strategy": ["2d"] * B,  # ADD THIS LINE
                "points1": [None] * B,                # ADD THIS LINE
                "points2": [None] * B,                # ADD THIS LINE
                "intrinsics1": [None] * B,            # ADD THIS LINE
                "intrinsics2": [None] * B,            # ADD THIS LINE
                "rotation1": [None] * B,              # ADD THIS LINE
                "rotation2": [None] * B,              # ADD THIS LINE
                "position1": [None] * B,              # ADD THIS LINE
                "position2": [None] * B,              # ADD THIS LINE
                "depth1": torch.ones((B, 1, dino_feat_1_0.shape[2], dino_feat_1_0.shape[3]), device=dino_feat_1_0.device),  # ADD THIS LINE
                "depth2": torch.ones((B, 1, dino_feat_1_0.shape[2], dino_feat_1_0.shape[3]), device=dino_feat_1_0.device),  # ADD THIS LINE
                }

        # Feature registration
        print("🔄 Applying feature registration...")
        dino_feat_1_0, dino_feat_2_0 = self.feature_register(batch_info, dino_feat_1_0, dino_feat_2_0)
        print(f"Registered features - img1: {dino_feat_1_0.shape}, img2: {dino_feat_2_0.shape}")

        # Cross attention
        print("🔄 Applying cross attention...")
        x_1_0, x_2_0 = self.apply_cross_attention(self, dino_feat_1_0, dino_feat_2_0, self.ca0s)
        print(f"Cross attention output - x1: {x_1_0.shape}, x2: {x_2_0.shape}")

        # Decoder
        print("🔄 Decoding features...")
        x = torch.concatenate([x_1_0, x_2_0], dim=1)  # (batch, 2*768, H, W)
        print(f"Concatenated features: {x.shape}")
        
        x = self.relu(self.conv1(x))  # (batch, 768, H, W)
        print(f"After conv1: {x.shape}")
        
        x = self.relu(self.conv2(x))  # (batch, 2, H, W)
        print(f"After conv2: {x.shape}")
        
        x = self.upsample(x)  # (batch, 2, target_H, target_W)
        print(f"After upsample: {x.shape}")
        
        x = x.permute(0, 2, 3, 1)  # (batch, target_H, target_W, 2)
        print(f"Final output: {x.shape}")
        
        return x

    def get_trainable_parameters(self):
        """Get statistics of trainable parameters"""
        total_params = 0
        trainable_params = 0
        grounding_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'grounding_adapter' in name:
                    grounding_params += param.numel()
        
        print(f"📊 Parameter Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   GroundingDINO adapter parameters: {grounding_params:,}")
        print(f"   Trainable ratio: {trainable_params/total_params*100:.1f}%")
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'grounding': grounding_params
        }


class VLCMUChangeDetectionDataset(Dataset):
    """
    VL-CMU Change Detection Dataset Loader
    
    Text file format:
    001_1_00_0.png 001_1_00_0.png
    1. White van
    
    002_1_00_0.png 002_1_00_0.png
    1. Cardboard boxes
    2. Blue bag
    """
    
    def __init__(self, root_dir, split='train', image_size=(504, 504), transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        
        # Paths
        self.split_dir = self.root_dir / split
        self.t0_dir = self.split_dir / 't0'
        self.t1_dir = self.split_dir / 't1'
        self.mask_dir = self.split_dir / 'mask'
        self.text_file = self.split_dir / f'text_cmu_{split}.txt'
        
        # Check if directories exist
        self._check_dataset_structure()
        
        # Default transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        
        # Load dataset samples
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples from VL-CMU-CD {split} split")
        
    def _check_dataset_structure(self):
        """Check if the dataset structure is correct"""
        required_dirs = [self.t0_dir, self.t1_dir, self.mask_dir]
        required_files = [self.text_file]
        
        missing_items = []
        for dir_path in required_dirs:
            if not dir_path.exists():
                missing_items.append(f"Directory: {dir_path}")
                
        for file_path in required_files:
            if not file_path.exists():
                missing_items.append(f"File: {file_path}")
        
        if missing_items:
            print(f"Warning: Missing items in dataset structure:")
            for item in missing_items:
                print(f"  - {item}")
        else:
            print(f"✓ VL-CMU-CD dataset structure verified for {self.split} split")
                
    def _load_samples(self):
        """Load samples from the text file with the specific format"""
        samples = []
        
        if not self.text_file.exists():
            print(f"Warning: Text file not found: {self.text_file}")
            # Create dummy samples for testing
            for i in range(10):
                samples.append({
                    'img1_path': f'dummy_t0_{i:03d}.png',
                    'img2_path': f'dummy_t1_{i:03d}.png',
                    'mask_path': f'dummy_mask_{i:03d}.png',
                    'caption': f'dummy caption {i}',
                    'id': f'{i:03d}'
                })
            return samples
        
        with open(self.text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check if this line contains image names (has .png)
            if '.png' in line:
                try:
                    # Parse image names line: img1_name img2_name
                    parts = line.split()
                    if len(parts) >= 2:
                        img1_name = parts[0]
                        img2_name = parts[1]
                        
                        # Construct full paths
                        img1_path = self.t0_dir / img1_name
                        img2_path = self.t1_dir / img2_name
                        mask_path = self.mask_dir / img1_name  # Assuming mask uses same name as img1
                        
                        # Collect caption lines
                        caption_lines = []
                        j = i + 1
                        while j < len(lines):
                            caption_line = lines[j].strip()
                            if not caption_line:
                                j += 1
                                continue
                            # If next line contains .png, it's a new sample
                            if '.png' in caption_line:
                                break
                            # If line starts with a number and dot, it's a caption
                            if re.match(r'^\d+\.\s+', caption_line):
                                # Remove the number prefix (e.g., "1. " -> "")
                                clean_caption = re.sub(r'^\d+\.\s+', '', caption_line)
                                caption_lines.append(clean_caption)
                            j += 1
                        
                        # Combine all caption lines
                        caption = ' '.join(caption_lines) if caption_lines else 'no description'
                        
                        samples.append({
                            'img1_path': str(img1_path),
                            'img2_path': str(img2_path),
                            'mask_path': str(mask_path),
                            'caption': caption.strip(),
                            'id': img1_name.split('.')[0]
                        })
                        
                        i = j  # Move to next sample
                    else:
                        i += 1
                        
                except Exception as e:
                    print(f"Error parsing line {i+1}: {line} - {e}")
                    i += 1
            else:
                i += 1
                    
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Handle dummy data
        if 'dummy' in sample['img1_path']:
            img1 = torch.randn(3, *self.image_size)
            img2 = torch.randn(3, *self.image_size)
            mask = torch.randint(0, 2, self.image_size)
            caption = sample['caption']
            return img1, img2, caption, mask
        
        try:
            # Load real images
            img1 = Image.open(sample['img1_path']).convert('RGB')
            img2 = Image.open(sample['img2_path']).convert('RGB')
            mask = Image.open(sample['mask_path']).convert('L')
            
            # Apply transforms
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).long().squeeze(0)
            
            caption = sample['caption']
            
            return img1, img2, caption, mask
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Fallback to dummy data
            img1 = torch.randn(3, *self.image_size)
            img2 = torch.randn(3, *self.image_size)
            mask = torch.randint(0, 2, self.image_size)
            caption = "error loading"
            
            return img1, img2, caption, mask


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    for batch_idx, (img1, img2, captions, targets) in enumerate(pbar):
        img1, img2, targets = img1.to(device), img2.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        try:
            # Forward pass
            outputs = model(img1, img2, captions)
            
            # Reshape outputs for loss calculation
            outputs = outputs.view(-1, 2)  # (batch*H*W, 2)
            targets = targets.view(-1)     # (batch*H*W,)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (img1, img2, captions, targets) in enumerate(tqdm(dataloader, desc='Validation')):
            img1, img2, targets = img1.to(device), img2.to(device), targets.to(device)
            
            try:
                outputs = model(img1, img2, captions)
                
                # Reshape for loss calculation
                outputs = outputs.view(-1, 2)
                targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train CrossAttention model on VL-CMU-CD dataset')
    parser.add_argument('--data_dir', type=str, default='/scratch/zl4701/datasets/VL-CMU-CD', 
                       help='Path to VL-CMU-CD dataset root')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
    
    # Model configuration arguments
    parser.add_argument('--dino_model', type=str, default='dinov2_vitb14', 
                       choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                       help='DINOv2 model variant')
    parser.add_argument('--layer', type=int, default=11, help='DINOv2 layer to extract features from')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_blocks', type=int, default=1, help='Number of cross-attention blocks')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    loss_tracker = LossTracker(args.save_dir)

    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Check GPU memory and give recommendations
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f'GPU memory: {gpu_memory:.1f} GB')
        
        # Validate model choice against available memory
        memory_requirements = {
            'dinov2_vits14': 6,
            'dinov2_vitb14': 10,
            'dinov2_vitl14': 18,
            'dinov2_vitg14': 32
        }
        
        required_memory = memory_requirements.get(args.dino_model, 10)
        if gpu_memory < required_memory:
            logger.warning(f'GPU memory ({gpu_memory:.1f}GB) may be insufficient for {args.dino_model} '
                         f'(requires ~{required_memory}GB). Consider using a smaller model.')
    
    # Validate layer number
    max_layers = {
        'dinov2_vits14': 11,
        'dinov2_vitb14': 11, 
        'dinov2_vitl14': 23,
        'dinov2_vitg14': 39
    }
    
    if args.layer > max_layers[args.dino_model]:
        logger.error(f'Layer {args.layer} is invalid for {args.dino_model}. '
                    f'Maximum layer is {max_layers[args.dino_model]}')
        return
    
    # Load DINOv2 backbone using torch.hub
    logger.info(f'Loading DINOv2 backbone: {args.dino_model}')
    dino = torch.hub.load(
        "facebookresearch/dinov2", args.dino_model, pretrained=True
    ).to(device)
    logger.info(f'DINOv2 model loaded successfully')
    
    # Initialize model with DINOv2-GroundingDINO adapter
    logger.info(f'Initializing CrossAttention model with layer {args.layer}')
    model = CrossAttention(
        dino1=dino,  # Pass the pre-loaded model instead of string
        layer1=args.layer,
        facet1="query",
        num_heads=args.num_heads,
        dropout_rate=0.1,
        target_shp=(504, 504),
        num_blocks=args.num_blocks
    )
    
    model = model.to(device)
    
    # 🔥 Get detailed parameter statistics
    param_stats = model.get_trainable_parameters()
    logger.info(f'Total parameters: {param_stats["total"]:,}')
    logger.info(f'Trainable parameters: {param_stats["trainable"]:,}')
    logger.info(f'GroundingDINO adapter parameters: {param_stats["grounding"]:,}')
    
    # Setup VL-CMU-CD datasets and dataloaders
    logger.info(f'Loading VL-CMU-CD dataset from: {args.data_dir}')
    
    train_dataset = VLCMUChangeDetectionDataset(
        root_dir=args.data_dir,
        split='train',
        image_size=(504, 504)
    )
    
    test_dataset = VLCMUChangeDetectionDataset(
        root_dir=args.data_dir,
        split='test',
        image_size=(504, 504)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 for debugging
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        test_dataset,  # Use test split for validation
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    logger.info(f'Train dataset: {len(train_dataset)} samples, {len(train_loader)} batches')
    logger.info(f'Test dataset: {len(test_dataset)} samples, {len(val_loader)} batches')
    
    # Test dataset loading
    if len(train_dataset) > 0:
        try:
            sample_img1, sample_img2, sample_caption, sample_mask = train_dataset[0]
            logger.info(f'Sample data shapes - img1: {sample_img1.shape}, img2: {sample_img2.shape}, mask: {sample_mask.shape}')
            logger.info(f'Sample caption: "{sample_caption}"')
            logger.info(f'Mask unique values: {torch.unique(sample_mask)}')
        except Exception as e:
            logger.warning(f'Error loading sample data: {e}')
    
    # Setup optimizer and loss function
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Resume training if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            logger.info(f'Resumed training from epoch {start_epoch}')
        except Exception as e:
            logger.warning(f'Failed to load checkpoint: {e}')
    
    # Training loop
    logger.info('Starting training...')
    for epoch in range(start_epoch, args.epochs):
        logger.info(f'Epoch {epoch+1}/{args.epochs}')
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        logger.info(f'Train Loss: {train_loss:.4f}')
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        logger.info(f'Val Loss: {val_loss:.4f}')

        # 记录损失
        loss_tracker.update(epoch, train_loss, val_loss)
        loss_tracker.save_losses()
        loss_tracker.plot_losses()
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'args': args,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest.pth'))
        logger.info(f'Saved latest checkpoint')
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, os.path.join(args.save_dir, 'best.pth'))
            logger.info(f'New best model saved with val_loss: {val_loss:.4f}')
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 保存测试结果图片
    print("🖼️ Saving test prediction images...")
    save_prediction_images(model, val_loader, device, 
                        os.path.join(args.save_dir, 'test_predictions'), num_samples=10)
    
    logger.info('Training completed!')
    logger.info(f'Best validation loss: {best_val_loss:.4f}')


def test_dataset_only():
    """Test function to verify VL-CMU-CD dataset loading without training"""
    root_dir = "/scratch/zl4701/datasets/VL-CMU-CD"
    
    print("Testing VL-CMU-CD Dataset Loading...")
    print("="*50)
    
    try:
        # Test train split
        print("Loading train dataset...")
        train_dataset = VLCMUChangeDetectionDataset(root_dir, split='train')
        print(f"Train dataset size: {len(train_dataset)}")
        
        # Test first few samples
        for i in range(min(3, len(train_dataset))):
            img1, img2, caption, mask = train_dataset[i]
            print(f"\nSample {i}:")
            print(f"  img1 shape: {img1.shape}")
            print(f"  img2 shape: {img2.shape}")
            print(f"  mask shape: {mask.shape}")
            print(f"  caption: '{caption}'")
            print(f"  mask unique values: {torch.unique(mask)}")
        
        # Test test split
        print("\nLoading test dataset...")
        test_dataset = VLCMUChangeDetectionDataset(root_dir, split='test')
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Test DataLoader
        print("\nTesting DataLoader...")
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)
        
        for batch_idx, (img1_batch, img2_batch, captions_batch, mask_batch) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  img1 batch shape: {img1_batch.shape}")
            print(f"  img2 batch shape: {img2_batch.shape}")
            print(f"  mask batch shape: {mask_batch.shape}")
            print(f"  captions: {captions_batch}")
            if batch_idx >= 1:  # Only test first 2 batches
                break
        
        print("\n✓ Dataset loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Dataset loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dinov2_grounding_integration():
    """Test DINOv2-GroundingDINO integration"""
    print("🧪 Testing DINOv2-GroundingDINO Integration")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load DINOv2
        print("📥 Loading DINOv2...")
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True).to(device)
        print("✓ DINOv2 loaded")
        
        # Create model
        print("🏗️ Creating CrossAttention model...")
        model = CrossAttention(
            dino1=dino,
            layer1=11, 
            facet1="query",
            num_heads=8,
            dropout_rate=0.1,
            target_shp=(504, 504),
            num_blocks=1
        ).to(device)
        print("✓ Model created")
        
        # Get parameter statistics
        param_stats = model.get_trainable_parameters()
        
        # Test forward pass
        print("🔄 Testing forward pass...")
        batch_size = 2
        img1 = torch.randn(batch_size, 3, 504, 504).to(device)
        img2 = torch.randn(batch_size, 3, 504, 504).to(device)
        captions = ["a red car on the road", "a blue truck in parking lot"]
        
        print(f"Input shapes: img1={img1.shape}, img2={img2.shape}")
        print(f"Captions: {captions}")
        
        # Forward pass
        with torch.no_grad():
            output = model(img1, img2, captions)
            
        print(f"✓ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output device: {output.device}")
        
        # Validate output
        expected_shape = (batch_size, 504, 504, 2)
        if output.shape == expected_shape:
            print(f"✓ Output shape correct: {output.shape}")
        else:
            print(f"⚠ Output shape mismatch: expected {expected_shape}, got {output.shape}")
        
        print("\n🎉 Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_enhancement():
    """Test text enhancement functionality specifically"""
    print("🧪 Testing Text Enhancement Functionality")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load minimal model for testing
        print("📥 Loading DINOv2...")
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True).to(device)
        
        # Create model
        model = CrossAttention(
            dino1=dino,
            layer1=11,
            facet1="query",
            num_heads=8,
            dropout_rate=0.1,
            target_shp=(504, 504),
            num_blocks=1
        ).to(device)
        
        # Test text enhancement directly
        print("🔄 Testing text enhancement...")
        batch_size = 1
        test_features = torch.randn(batch_size, 768, 36, 36).to(device)
        test_captions = ["a red car driving on a highway"]
        
        print(f"Input features shape: {test_features.shape}")
        print(f"Test captions: {test_captions}")
        
        # Test enhancement
        enhanced_features = model.enhance_features_with_text(test_features, test_captions)
        
        print(f"✓ Text enhancement successful!")
        print(f"Enhanced features shape: {enhanced_features.shape}")
        
        # Validate enhancement
        if enhanced_features.shape == test_features.shape:
            print("✓ Feature shapes match")
        else:
            print(f"⚠ Shape mismatch: {enhanced_features.shape} vs {test_features.shape}")
        
        # Test with empty captions
        print("\n🔄 Testing with empty captions...")
        empty_enhanced = model.enhance_features_with_text(test_features, ["", "  "])
        print(f"Empty caption result shape: {empty_enhanced.shape}")
        
        print("\n🎉 Text enhancement test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Text enhancement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import sys
    
    # Command line options for different tests
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test-dataset':
            test_dataset_only()
        elif sys.argv[1] == '--test-integration':
            test_dinov2_grounding_integration()
        elif sys.argv[1] == '--test-text':
            test_text_enhancement()
        else:
            main()
    else:
        main()


# Usage examples are below:
"""
# Test dataset loading only:
python train_crossattention_integrated.py --test-dataset

# Test DINOv2-GroundingDINO integration:
python train_crossattention_integrated.py --test-integration

# Test text enhancement specifically:
python train_crossattention_integrated.py --test-text

# Run training with default settings:
python train_crossattention_withwarp.py --data_dir /scratch/zl4701/datasets/VL-CMU-CD

# Run training with custom settings:
python train_crossattention_integrated.py \
    --data_dir /scratch/zl4701/datasets/VL-CMU-CD \
    --dino_model dinov2_vitb14 \
    --layer 11 \
    --batch_size 4 \
    --epochs 50 \
    --lr 1e-4 \
    --num_heads 8 \
    --num_blocks 2

# Resume training from checkpoint:
python train_crossattention_integrated.py \
    --data_dir /scratch/zl4701/datasets/VL-CMU-CD \
    --resume ./checkpoints/best.pth

# Train on GPU (when available):
srun --gres=gpu:1 --time=4:00:00 --pty bash
conda activate /scratch/zl4701/conda-envs/unified-env
python train_crossattention_integrated.py --data_dir /scratch/zl4701/datasets/VL-CMU-CD --epochs 20
"""
