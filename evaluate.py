import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from preprocess_data import get_data_loaders
from da_vit_model import DAViT

def evaluate_model(model_path='best_model.pth', batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    _, val_loader = get_data_loaders(batch_size=batch_size)

    model = DAViT(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Metrics
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=[
        'NonDemented', 'VeryMild', 'Mild', 'Moderate'
    ]))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['Non', 'VM', 'M', 'Mod'], yticklabels=['Non', 'VM', 'M', 'Mod'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    # ROC-AUC
    y_true = np.eye(4)[all_labels]
    y_score = np.array(all_probs)
    auc_score = roc_auc_score(y_true, y_score, multi_class='ovr')
    print(f"Macro ROC-AUC Score: {auc_score:.4f}")

    for i in range(4):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f'Class {i}')

    plt.title("ROC Curves by Class")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig("roc_curves.png")
    plt.show()

if __name__ == '__main__':
    evaluate_model()
