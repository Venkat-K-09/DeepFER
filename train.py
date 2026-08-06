import os
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.models import build_model
from src.utils import calculate_metrics, plot_training_history, save_model

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    total = len(all_targets)
    epoch_loss = running_loss / total
    metrics = calculate_metrics(all_targets, all_preds)
    return epoch_loss, metrics, all_targets, all_preds

def main():
    parser = argparse.ArgumentParser(description="Train DeepFER Model")
    parser.add_argument('--model', type=str, default='custom', choices=['custom', 'transfer'], help="Model type")
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'efficientnet', 'mobilenet'], help="Backbone for transfer model")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--use_weights', action='store_true', help="Use class weights in loss function")
    parser.add_argument('--save_name', type=str, default=None, help="Name tag for saved model")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loaders
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        batch_size=args.batch_size, num_workers=0
    )

    # Loss criterion
    if args.use_weights:
        weights_tensor = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        print(f"Using class-weighted CrossEntropyLoss: {weights_tensor}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Build model
    model = build_model(model_type=args.model, backbone_name=args.backbone).to(device)
    save_tag = args.save_name or (f"{args.model}_{args.backbone}" if args.model == 'transfer' else 'custom_cnn')

    print(f"--- Starting Training: {save_tag.upper()} ---")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    best_val_f1 = 0.0
    best_model_path = os.path.join('models', f"best_{save_tag}.pth")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['macro_f1']

        scheduler.step(val_f1)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        epoch_time = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            metadata = {
                'epoch': epoch,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'model_type': args.model,
                'backbone': args.backbone,
                'save_tag': save_tag
            }
            save_model(model, best_model_path, metadata=metadata)

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time/60:.2f} minutes! Best Val Macro F1: {best_val_f1:.4f}")

    # Plot and save curves
    plot_training_history(history, save_path=os.path.join('models', f"curves_{save_tag}.png"))
    
    with open(os.path.join('models', f"history_{save_tag}.json"), 'w') as f:
        json.dump(history, f, indent=4)

if __name__ == '__main__':
    main()
