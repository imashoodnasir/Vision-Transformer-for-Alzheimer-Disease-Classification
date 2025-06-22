import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableMHSA(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, patch_grid=14):
        super(DeformableMHSA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.patch_grid = patch_grid

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Offset learning (x, y) per head per location
        self.offsets = nn.Parameter(torch.zeros(1, num_heads, patch_grid * patch_grid, 2))

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, N, H, D_head)

        q = q.permute(0, 2, 1, 3)  # (B, H, N, D_head)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Apply deformable attention via offset sampling
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn_weights = attn_scores.softmax(dim=-1)

        # Deformable logic can be extended using dynamic position sampling
        # Placeholder: use standard attention weights
        out = attn_weights @ v  # (B, H, N, D_head)

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)
