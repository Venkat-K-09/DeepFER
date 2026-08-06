import os
import json
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch

from src.dataset import EMOTION_CLASSES

def calculate_metrics(y_true, y_pred):
    """
    Computes classification metrics: accuracy, precision, recall, f1-score.
    """
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per class metrics
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    per_class = {}
    for idx, cls_name in enumerate(EMOTION_CLASSES):
        per_class[cls_name] = {
            'precision': float(p_class[idx]),
            'recall': float(r_class[idx]),
            'f1_score': float(f1_class[idx])
        }

    return {
        'accuracy': float(acc),
        'macro_precision': float(p_macro),
        'macro_recall': float(r_macro),
        'macro_f1': float(f1_macro),
        'weighted_precision': float(p_weighted),
        'weighted_recall': float(r_weighted),
        'weighted_f1': float(f1_weighted),
        'per_class': per_class
    }

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png', title='Confusion Matrix'):
    """
    Plots and saves a styled seaborn confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set_theme(style="dark")
    
    # Annotate with count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm, annot=annot, fmt='', cmap='Blues',
        xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES,
        cbar=True, ax=ax, linewidths=0.5
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Emotion', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_training_history(history, save_path='training_curves.png'):
    """
    Plots training and validation loss & accuracy over epochs.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss')
    ax1.set_title('Loss vs Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Acc')
    ax2.set_title('Accuracy vs Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {save_path}")

def save_model(model, path, metadata=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state = {
        'state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    torch.save(state, path)
    print(f"Model checkpoint saved to {path}")
