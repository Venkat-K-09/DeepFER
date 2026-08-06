import os
from pathlib import Path
from typing import Tuple, List, Dict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(EMOTION_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(EMOTION_CLASSES)}

class FERDataset(Dataset):
    """
    Facial Emotion Recognition Dataset class with fast NumPy memory caching.
    """
    def __init__(self, root_dir: str, transform=None, use_cache: bool = True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_cache = use_cache
        self.samples = []
        self.labels = []
        self.images_cache = []
        self.class_counts = {cls_name: 0 for cls_name in EMOTION_CLASSES}

        for cls_name in EMOTION_CLASSES:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    label = CLASS_TO_IDX[cls_name]
                    self.samples.append(str(img_path))
                    self.labels.append(label)
                    self.class_counts[cls_name] += 1

        if self.use_cache:
            for path in self.samples:
                img = cv2.imread(path)
                if img is None:
                    img = np.zeros((48, 48, 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.images_cache.append(img)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_cache and len(self.images_cache) > idx:
            img_np = self.images_cache[idx]
            pil_img = Image.fromarray(img_np)
        else:
            path = self.samples[idx]
            pil_img = Image.open(path).convert('RGB')
        
        label = self.labels[idx]

        if self.transform:
            image_tensor = self.transform(pil_img)
        else:
            image_tensor = transforms.ToTensor()(pil_img)

        return image_tensor, label

    def get_class_weights(self) -> torch.Tensor:
        counts = [self.class_counts[cls_name] for cls_name in EMOTION_CLASSES]
        total = sum(counts)
        num_classes = len(EMOTION_CLASSES)
        
        weights = []
        for c in counts:
            if c > 0:
                weights.append(total / (num_classes * c))
            else:
                weights.append(1.0)
                
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        weights_tensor = weights_tensor / weights_tensor.sum() * num_classes
        return weights_tensor

def get_transforms(img_size: int = 48, is_train: bool = True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(img_size, padding=4, padding_mode='edge'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders(
    train_dir: str = 'dataset/train',
    test_dir: str = 'dataset/test',
    batch_size: int = 64,
    img_size: int = 48,
    val_split: float = 0.1,
    num_workers: int = 0,
    use_cache: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    
    train_transform = get_transforms(img_size=img_size, is_train=True)
    eval_transform = get_transforms(img_size=img_size, is_train=False)

    full_train_dataset = FERDataset(train_dir, transform=train_transform, use_cache=use_cache)
    test_dataset = FERDataset(test_dir, transform=eval_transform, use_cache=use_cache)

    class_weights = full_train_dataset.get_class_weights()

    total_size = len(full_train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_weights

if __name__ == '__main__':
    train_loader, val_loader, test_loader, weights = get_dataloaders(num_workers=0, use_cache=False)
    print("Dataloaders initialized successfully!")
