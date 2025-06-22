import torch
import torch.nn as nn
from vit_patch_embedding import PatchEmbedding
from deformable_mhsa import DeformableMHSA

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = DeformableMHSA(embed_dim=embed_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # Deformable MHSA with residual
        x = x + self.mlp(self.norm2(x))   # Feedforward + residual
        return x

class DAViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=4,
                 embed_dim=768, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(DAViT, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)       # (B, N, D)
        x = self.blocks(x)            # Transformer stack
        x = self.norm(x)
        x = x.mean(dim=1)             # Global average pooling
        return self.classifier(x)     # (B, num_classes)
