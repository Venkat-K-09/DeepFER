import os
import argparse
import json
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import get_dataloaders, EMOTION_CLASSES
from src.models import build_model
from src.utils import calculate_metrics, plot_confusion_matrix

def evaluate_model(model_path, model_type='custom', backbone_name='resnet18', batch_size=64, save_tag='eval'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating model on device: {device}")

    _, _, test_loader, _ = get_dataloaders(batch_size=batch_size, num_workers=0)

    model = build_model(model_type=model_type, backbone_name=backbone_name).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    metadata = checkpoint.get('metadata', {})
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model checkpoint from: {model_path}")
    if metadata:
        print(f"Checkpoint Metadata: {metadata}")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_targets, all_preds)

    print("\n" + "="*50)
    print(f" TEST EVALUATION RESULTS: {save_tag.upper()} ")
    print("="*50)
    print(f"Overall Test Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision:          {metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:             {metrics['macro_recall']*100:.2f}%")
    print(f"Macro F1-Score:           {metrics['macro_f1']*100:.2f}%")
    print(f"Weighted F1-Score:        {metrics['weighted_f1']*100:.2f}%")
    print("-" * 50)
    print("PER-CLASS PERFORMANCE:")
    for cls_name, vals in metrics['per_class'].items():
        print(f"  {cls_name.capitalize():<10} | Precision: {vals['precision']*100:5.1f}% | Recall: {vals['recall']*100:5.1f}% | F1: {vals['f1_score']*100:5.1f}%")
    print("="*50)

    # Plot Confusion Matrix
    cm_path = os.path.join('models', f"confusion_matrix_{save_tag}.png")
    plot_confusion_matrix(all_targets, all_preds, save_path=cm_path, title=f"Confusion Matrix ({save_tag.upper()})")

    # Save evaluation report JSON
    report_path = os.path.join('models', f"evaluation_{save_tag}.json")
    report_data = {
        'model_path': model_path,
        'model_type': model_type,
        'backbone': backbone_name,
        'save_tag': save_tag,
        'metrics': metrics
    }
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=4)
    print(f"Evaluation report saved to {report_path}")

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepFER Model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to saved model checkpoint (.pth)")
    parser.add_argument('--model', type=str, default='custom', choices=['custom', 'transfer'], help="Model type")
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'efficientnet', 'mobilenet'], help="Backbone for transfer model")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--save_name', type=str, default='test_eval', help="Tag for output files")

    args = parser.parse_args()
    evaluate_model(
        model_path=args.model_path,
        model_type=args.model,
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        save_tag=args.save_name
    )

if __name__ == '__main__':
    main()
