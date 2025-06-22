import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        # Input shape: (B, C, H, W)
        x = self.projection(x)                      # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)            # (B, N, D)
        x = x + self.position_embeddings            # Add positional encoding
        return x                                    # Output: (B, N, D)
