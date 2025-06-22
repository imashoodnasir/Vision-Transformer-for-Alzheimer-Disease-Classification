import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from da_vit_model import DAViT
from preprocess_data import get_data_loaders
import torchvision.transforms as T

def generate_attention_map(image_tensor, model, layer_id=-1):
    model.eval()
    with torch.no_grad():
        features = model.patch_embed(image_tensor.unsqueeze(0))
        features = model.blocks[:layer_id](features)
        x = model.blocks[layer_id].norm1(features)
        attn_weights = model.blocks[layer_id].attn.attn_weights(x)  # (B, H, N, N)
        attn_map = attn_weights.mean(1).squeeze(0)  # (N, N)
        return attn_map

def visualize_attention(image_tensor, attn_map, save_path="attention_overlay.png"):
    h = w = int(np.sqrt(attn_map.shape[0]))
    attn_mask = attn_map.mean(0).reshape(h, w).cpu().numpy()
    attn_mask = cv2.resize(attn_mask, (224, 224))

    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * attn_mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * heatmap / 255 + 0.5 * image)  # Blending

    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Attention Overlay")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    model = DAViT(num_classes=4)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()

    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    _, val_loader = get_data_loaders(batch_size=1)
    image_tensor, label = next(iter(val_loader))
    image_tensor = image_tensor[0]

    attn_map = generate_attention_map(image_tensor, model)
    visualize_attention(image_tensor, attn_map, save_path="attention_overlay.png")

    print("✅ Attention visualization saved as attention_overlay.png")
