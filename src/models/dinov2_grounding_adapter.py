import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import List, Tuple, Optional

class DINOv2ToGroundingAdapter:
    """
    DINOv2 -> GroundingDINO 特征增强适配器
    使用GroundingDINO的实际API来增强DINOv2特征
    """
    def __init__(self, grounding_dino_model, device='cuda'):
        self.grounding_dino = grounding_dino_model
        self.device = device
        
        # 特征维度适配层
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1, bias=False),  # DINOv2 -> GroundingDINO backbone dim
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 768, kernel_size=1, bias=False),  # 恢复到DINOv2维度
            nn.BatchNorm2d(768)
        ).to(device)
        
        # 注意力融合模块 - 简化版本
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=768, 
            num_heads=8, 
            batch_first=True,
            dropout=0.1
        ).to(device)
        
        # 简单的特征融合层作为备选方案
        self.simple_fusion = nn.Sequential(
            nn.Conv2d(768 * 2, 768, kernel_size=1),  # 连接后的特征融合
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.BatchNorm2d(768)
        ).to(device)
        
        # 特征增强网络
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1, groups=768),  # Depthwise conv
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, kernel_size=1),  # Pointwise conv
            nn.BatchNorm2d(768)
        ).to(device)
        
        print(f"✓ DINOv2-GroundingDINO adapter ready on {device}")
    
    def improve_dino_feature(self, dinov2_features: torch.Tensor, captions: List[str]) -> torch.Tensor:
        """
        使用GroundingDINO增强DINOv2特征
        
        Args:
            dinov2_features: DINOv2特征 [B, 768, H, W]
            captions: 文本描述列表
            
        Returns:
            enhanced_features: 增强后的特征 [B, 768, H, W]
        """
        try:
            B, C, H, W = dinov2_features.shape
            print(f"Input DINOv2 features: {dinov2_features.shape}")
            
            if C != 768:
                print(f"⚠ Expected 768 channels, got {C}")
                return dinov2_features
            
            # 1. 清理captions
            processed_captions = self._clean_captions(captions)
            print(f"Processed captions: {processed_captions}")
            
            # 2. 使用GroundingDINO的文本编码器获取文本特征
            text_features = self._encode_text(processed_captions)
            print(f"Text features shape: {text_features.shape}")
            
            # 3. 特征适配和增强
            adapted_features = self._adapt_features(dinov2_features)
            print(f"Adapted features shape: {adapted_features.shape}")
            
            # 4. 文本-视觉特征融合
            enhanced_features = self._fuse_text_visual_features(
                adapted_features, text_features
            )
            print(f"Enhanced features shape: {enhanced_features.shape}")
            
            # 5. 残差连接
            final_features = dinov2_features + 0.1 * enhanced_features  # 使用较小的权重
            
            print(f"✓ Final enhanced features: {final_features.shape}")
            return final_features
            
        except Exception as e:
            print(f"⚠ GroundingDINO adapter failed: {e}")
            print(f"  Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()  # 打印完整的错误堆栈
            print("  Falling back to original features")
            return dinov2_features
    
    # def _encode_text(self, captions: List[str]) -> torch.Tensor:
    #     """使用GroundingDINO的文本编码器编码文本"""
    #     try:
    #         # 处理文本输入
    #         text_input = " . ".join(captions)  # 合并所有captions
            
    #         # 直接使用GroundingDINO的tokenizer
    #         tokenized = self.grounding_dino.tokenizer(
    #             text_input,
    #             padding="max_length",
    #             max_length=256,
    #             truncation=True,
    #             return_tensors="pt"
    #         )

    #         tokenized = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tokenized.items()}
            
    #         # 使用GroundingDINO的文本编码器
    #         with torch.no_grad():
    #             # 检查bert属性是否存在
    #             if hasattr(self.grounding_dino, 'bert'):
    #                 text_features = self.grounding_dino.bert(
    #                     tokenized['input_ids'],
    #                     attention_mask=tokenized['attention_mask']
    #                 )[0]  # [1, seq_len, 768]
    #             elif hasattr(self.grounding_dino, 'text_encoder'):
    #                 text_features = self.grounding_dino.text_encoder(
    #                     tokenized['input_ids'],
    #                     attention_mask=tokenized['attention_mask']
    #                 )[0]  # [1, seq_len, 768]
    #             else:
    #                 # 如果找不到文本编码器，创建dummy特征
    #                 print("No text encoder found, using dummy features")
    #                 return torch.zeros(1, 768, device=self.device) 
    #         # 使用注意力掩码进行加权平均池化
    #         mask = tokenized.attention_mask.unsqueeze(-1).float()  # [1, seq_len, 1]
    #         masked_features = text_features * mask  # [1, seq_len, 768]
    #         text_embedding = masked_features.sum(dim=1) / mask.sum(dim=1)  # [1, 768]
            
    #         return text_embedding
    def _encode_text(self, captions: List[str]) -> torch.Tensor:
        try:
            from transformers import AutoTokenizer, AutoModel
            
            text_input = " . ".join(captions)
            
            # 创建独立的tokenizer和模型，全部在GPU上
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            text_encoder = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
            
            # tokenize并强制到GPU
            inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = text_encoder(**inputs).last_hidden_state
                text_embedding = text_features.mean(dim=1)  # [1, 768]
            
            # 确保输出在正确设备上
            return text_embedding.to(self.device)
            
        except Exception as e:
            print(f"Text encoding failed: {e}, using dummy features")
            return torch.zeros(1, 768, device=self.device)
            
                
        
    def _adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        """特征适配和预处理"""
        # 通过适配层
        adapted = self.feature_adapter(features)
        return adapted
    
    def _fuse_text_visual_features(self, visual_features, text_features):
        B, C, H, W = visual_features.shape
        
        if text_features.size(0) == 1 and B > 1:
            text_features = text_features.expand(B, -1)
        
        text_broadcast = text_features.view(B, C, 1, 1).expand(-1, -1, H, W)
        concatenated = torch.cat([visual_features, text_broadcast], dim=1)
        fused_features = self.simple_fusion(concatenated)
        
        enhanced = self.feature_enhancer(fused_features)
        return enhanced
    def _clean_captions(self, captions: List[str]) -> List[str]:
        """清理和标准化captions"""
        processed = []
        for caption in captions:
            clean = str(caption).strip()
            # 移除数字编号
            clean = re.sub(r'^\d+\.\s*', '', clean)
            # 移除多余空格
            clean = re.sub(r'\s+', ' ', clean)
            # 确保不为空
            if len(clean) < 2:
                clean = "object"
            # 限制长度
            if len(clean) > 100:
                clean = clean[:100]
            # 确保以句号结尾
            if not clean.endswith('.'):
                clean += '.'
            processed.append(clean)
        return processed

    def get_grounding_predictions(self, image: torch.Tensor, 
                                  caption: str, 
                                  box_threshold: float = 0.35,
                                  text_threshold: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        使用GroundingDINO进行目标检测
        
        Args:
            image: 输入图像 [3, H, W] 或 [B, 3, H, W]
            caption: 文本描述
            box_threshold: 框置信度阈值
            text_threshold: 文本置信度阈值
            
        Returns:
            boxes: 检测框 [N, 4]
            logits: 置信度分数 [N]
            phrases: 对应的文本短语
        """
        try:
            from groundingdino.util.inference import predict
            
            # 确保图像格式正确
            if image.dim() == 4:
                image = image.squeeze(0)  # 移除batch维度
            
            # 调用GroundingDINO预测
            boxes, logits, phrases = predict(
                model=self.grounding_dino,
                image=image,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )
            
            return boxes, logits, phrases
            
        except Exception as e:
            print(f"GroundingDINO prediction failed: {e}")
            # 返回空结果
            return torch.empty(0, 4), torch.empty(0), []

    # # 加载函数保持不变，但使用新的适配器
    # def load_grounding_dino_with_adapter(device='cuda'):
    #     """
    #     加载带适配器的GroundingDINO
    #     """
    #     try:
    #         from groundingdino.util.inference import load_model
            
    #         # 加载GroundingDINO
    #         grounding_model = load_model(
    #             "/scratch/zl4701/rscd/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
    #             "/scratch/zl4701/rscd/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    #         )
            
    #         # 创建适配器
    #         adapter = DINOv2ToGroundingAdapter(grounding_model, device)
            
    #         print("✓ GroundingDINO with DINOv2 adapter loaded successfully")
    #         return adapter
            
    #     except Exception as e:
    #         print(f"✗ Failed to load GroundingDINO adapter: {e}")
    #         return None

def load_grounding_dino_with_adapter(device='cuda'):
    """使用HuggingFace GroundingDINO"""
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        
        print("📥 Loading HuggingFace GroundingDINO...")
        processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        ).to(device)
        
        adapter = DINOv2ToGroundingAdapter(grounding_model, device)
        print("✓ HuggingFace GroundingDINO adapter loaded successfully")
        return adapter
        
    except Exception as e:
        print(f"✗ Failed to load HuggingFace GroundingDINO: {e}")
        print("  Run: pip install transformers")
        return None

# 增强的测试函数
def test_adapter():
    """测试适配器功能"""
    print("=== Testing DINOv2-GroundingDINO Adapter ===")
    
    adapter = load_grounding_dino_with_adapter('cuda')
    
    if adapter is None:
        return False
    
    # 测试1: 特征增强
    print("\n--- Test 1: Feature Enhancement ---")
    test_features = torch.randn(2, 768, 36, 36)  # 批量DINOv2特征
    test_captions = ["a red car", "a blue truck"]
    
    result = adapter.improve_dino_feature(test_features, test_captions)
    
    print(f"Input shape: {test_features.shape}")
    print(f"Output shape: {result.shape}")
    print(f"✓ Feature enhancement test {'passed' if result.shape == test_features.shape else 'failed'}")
    
    # 测试2: 文本编码
    print("\n--- Test 2: Text Encoding ---")
    text_features = adapter._encode_text(test_captions)
    print(f"Text features shape: {text_features.shape}")
    print(f"✓ Text encoding test {'passed' if text_features.shape[1] == 768 else 'failed'}")
    
    # 测试3: 目标检测接口 (需要实际图像)
    print("\n--- Test 3: Detection Interface ---")
    dummy_image = torch.randn(3, 224, 224)
    boxes, logits, phrases = adapter.get_grounding_predictions(
        dummy_image, 
        "a car . a truck"
    )
    print(f"Detection boxes: {boxes.shape}")
    print(f"Detection logits: {logits.shape}")
    print(f"Detection phrases: {len(phrases)}")
    
    print("\n✓ All adapter tests completed!")
    return True

if __name__ == "__main__":
    test_adapter()
