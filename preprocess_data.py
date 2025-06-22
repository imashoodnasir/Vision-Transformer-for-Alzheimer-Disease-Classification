import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

# Define class names
CLASS_NAMES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
IMG_SIZE = 224

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize([0.5], [0.5])
])

class AlzheimerMRIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def load_dataset(root_dir='./data'):
    image_paths, labels = [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(root_dir, class_name)
        for img_file in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_file))
            labels.append(label_idx)

    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_dataset = AlzheimerMRIDataset(X_train, y_train, transform=transform)
    val_dataset = AlzheimerMRIDataset(X_val, y_val, transform=transform)

    return train_dataset, val_dataset

def get_data_loaders(batch_size=32):
    train_dataset, val_dataset = load_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
