import torch
import torch.nn as nn
from torch.optim import AdamW
from da_vit_model import DAViT
from preprocess_data import get_data_loaders
import os

def train_model(epochs=30, lr=3e-4, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Data
    train_loader, val_loader = get_data_loaders(batch_size=batch_size)

    # Model
    model = DAViT(num_classes=4)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    best_acc = 0.0
    save_path = './best_model.pth'

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Accuracy: {acc:.2f}%")

        # Validation
        model.eval()
        with torch.no_grad():
            val_correct, val_total = 0, 0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

            val_acc = 100 * val_correct / val_total
            print(f"Validation Accuracy: {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print("✅ Best model saved.")

if __name__ == '__main__':
    train_model()
