import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = self.pool(x)
        x = self.dropout(x)
        return x

class CustomFERCNN(nn.Module):
    """
    Custom Deep Convolutional Neural Network for Facial Emotion Recognition.
    Specially designed for 48x48 input images with residual feature aggregation.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.block1 = ConvBlock(3, 64, dropout_rate=0.25)
        self.block2 = ConvBlock(64, 128, dropout_rate=0.25)
        self.block3 = ConvBlock(128, 256, dropout_rate=0.3)
        self.block4 = ConvBlock(256, 512, dropout_rate=0.35)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

class TransferFERModel(nn.Module):
    """
    Transfer Learning Model for Facial Emotion Recognition leveraging pre-trained CNN backbones.
    Supports ResNet-18, EfficientNet-B0, MobileNet-V3.
    """
    def __init__(self, backbone_name='resnet18', num_classes=7, freeze_backbone=False):
        super().__init__()
        self.backbone_name = backbone_name.lower()

        if self.backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif self.backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif self.backbone_name == 'efficientnet':
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif self.backbone_name == 'mobilenet':
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.backbone = models.mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

def build_model(model_type='custom', backbone_name='resnet18', num_classes=7, freeze_backbone=False):
    if model_type.lower() == 'custom':
        return CustomFERCNN(num_classes=num_classes)
    elif model_type.lower() == 'transfer':
        return TransferFERModel(backbone_name=backbone_name, num_classes=num_classes, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

if __name__ == '__main__':
    dummy_input = torch.randn(4, 3, 48, 48)
    
    custom_model = build_model('custom')
    out_custom = custom_model(dummy_input)
    print(f"Custom CNN Output shape: {out_custom.shape}")

    transfer_model = build_model('transfer', backbone_name='resnet18')
    out_transfer = transfer_model(dummy_input)
    print(f"Transfer ResNet18 Output shape: {out_transfer.shape}")
