import torch
import torch.nn as nn
import torch.nn.functional as F

# Load the pretrained DINOv2 ViT-S/14 with registers model from torch.hub
dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

class CrossAttentionBlock(nn.Module):
    """Cross-attention block with feedforward layers, residual connections, and normalization."""
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        
        # Multi-head cross-attention layers
        self.cross_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward networks
        self.ffn_1 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.ffn_2 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # LayerNorm and Dropout for residual connections
        self.norm1_1 = nn.LayerNorm(d_model)
        self.norm1_2 = nn.LayerNorm(d_model)
        self.dropout1_1 = nn.Dropout(dropout)
        self.dropout1_2 = nn.Dropout(dropout)

        self.norm2_1 = nn.LayerNorm(d_model)
        self.norm2_2 = nn.LayerNorm(d_model)
        self.dropout2_1 = nn.Dropout(dropout)
        self.dropout2_2 = nn.Dropout(dropout)

    def forward(self, feat1, feat2):
        # Cross-attention from feat1 to feat2
        attn_output1, _ = self.cross_attn_1(feat1, feat2, feat2)
        feat1 = feat1 + self.dropout1_1(attn_output1)
        feat1 = self.norm1_1(feat1)
        
        # Feedforward network for feat1
        ffn_output1 = self.ffn_1(feat1)
        feat1 = feat1 + self.dropout1_2(ffn_output1)
        feat1 = self.norm1_2(feat1)
        
        # Cross-attention from feat2 to feat1
        attn_output2, _ = self.cross_attn_2(feat2, feat1, feat1)
        feat2 = feat2 + self.dropout2_1(attn_output2)
        feat2 = self.norm2_1(feat2)
        
        # Feedforward network for feat2
        ffn_output2 = self.ffn_2(feat2)
        feat2 = feat2 + self.dropout2_2(ffn_output2)
        feat2 = self.norm2_2(feat2)

        return feat1, feat2

class ChangeDetectionNet(nn.Module):
    def __init__(self, backbone_output_dim=384):
        super(ChangeDetectionNet, self).__init__()
        self.backbone = dinov2_vits14_reg
        self.cross_attention_block = CrossAttentionBlock(d_model=backbone_output_dim, nhead=8, dim_feedforward=1024)
        self.conv1 = nn.Conv2d(backbone_output_dim * 2, backbone_output_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(backbone_output_dim, 2, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=14, mode='bilinear', align_corners=False)

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, img_t0, img_t1):
        # Extract patch token features from both images
        with torch.no_grad():
            features_t0 = self.backbone.forward_features(img_t0)
            features_t1 = self.backbone.forward_features(img_t1)
            F0 = features_t0["x_norm_patchtokens"]
            F1 = features_t1["x_norm_patchtokens"]
        # print(F0.shape,F1.shape)
        # torch.Size([1, 1296, 384]) torch.Size([1, 1296, 384])
        # Reshape patch tokens to (batch, embedding_dim, height, width)
        b, n, c = F0.shape
        h = w = int(n ** 0.5)
        F0 = F0.permute(0, 2, 1).reshape(b, c, h, w)
        F1 = F1.permute(0, 2, 1).reshape(b, c, h, w)
        # print(F0.shape,F1.shape)
        # torch.Size([1, 384, 36, 36]) torch.Size([1, 384, 36, 36])
        # Apply cross-attention block
        attn_F0, attn_F1 = self.cross_attention_block(F0.reshape(b, n, c), F1.reshape(b, n, c))
        # print(attn_F0.shape, attn_F1.shape)
        # torch.Size([1, 1296, 384]) torch.Size([1, 1296, 384])
        # Concatenate features and decode
        concatenated_features = torch.cat([attn_F0.permute(0, 2, 1).reshape(b, c, h, w),
                                           attn_F1.permute(0, 2, 1).reshape(b, c, h, w)], dim=1)
        # print(concatenated_features.shape)
        # torch.Size([1, 768, 36, 36])
        x = F.relu(self.conv1(concatenated_features))
        x = self.conv2(x)

        # Upsample and apply softmax activation
        change_mask = self.upsample(x)
        # torch.Size([1, 2, H, W]): 1st probability pixel no change; 2nd: probability pixel changed
        change_mask = F.softmax(change_mask, dim=1)
        # 2 channel add to 1
        # print(change_mask.shape)
        # torch.Size([1, 2, 504, 504])
        return change_mask


# Example usage:
# model = ChangeDetectionNet(backbone_output_dim=384)
# img_t0 = torch.randn(1, 3, 504, 504)
# img_t1 = torch.randn(1, 3, 504, 504)
# output = model(img_t0, img_t1)
# print("Output shape:", output.shape)  # Expected shape: (batch_size, 2, H, W)
