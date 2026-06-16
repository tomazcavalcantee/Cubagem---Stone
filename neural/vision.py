import time
import torch
import logging

import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import DenseNet121_Weights, DenseNet121_Weights

class ProjectionHead(nn.Module):
    """
    Projection head with two linear layers and ReLU activation
    """
    def __init__(self, input_dim, hidden_dim=2048, out_dim=128):
        super(ProjectionHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # <--- corrigido aqui
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.head(x)


class Backbone(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True, num_classes=10, freeze_until: str = 'all'):
        """
        Initializes the backbone model with a projection head and optional classifier.
        Args:
            backbone_name (str): Name of the backbone model to use ('resnet18', 'densenet121', 'vit_small', 'vit_base', 'dinov2').
            pretrained (bool): Whether to use pretrained weights.
            num_classes (int): Number of output classes for the classifier.
            freeze_until (str): Parameter name until which to freeze the backbone layers.
        """

        super(Backbone, self).__init__()

        self.backbone_name = backbone_name.lower() if backbone_name else None

        if self.backbone_name == 'resnet18':
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            input_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()

        elif self.backbone_name == 'densenet121':
            backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
            input_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()

        elif self.backbone_name == 'vit_small':
            backbone = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
            input_dim = backbone.head.in_features
            backbone.head = nn.Identity()

        elif self.backbone_name == 'vit_base':
            backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
            input_dim = backbone.head.in_features
            backbone.head = nn.Identity()

        elif self.backbone_name == 'dinov2':
            backbone = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14')
            input_dim = backbone.embed_dim

        else:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. "\
                "Choose from 'resnet18', 'densenet121', 'vit_small', 'vit_base'," \
                " dinov2', 'labnet', 'vehiclenetrevisited', or 'vehiclenet'.")

        # Ensure freeze_until is a string or None
        if not isinstance(freeze_until, str):
            raise ValueError("freeze_until must be a string")
        
        # Ensure freeze_until is a parameter name in the backbone
        if freeze_until not in dict(backbone.named_parameters()) and freeze_until != 'all' and freeze_until.lower() != 'none':
            print(f"Available parameters in {self.backbone_name}:")
            for name, _ in backbone.named_parameters():
                print(name)
            raise ValueError(f"freeze_until '{freeze_until}' is not a valid parameter name in the backbone. Neither 'all'.")
    
        # Selective freezing based on parameter names
        if freeze_until not in ['all', 'none']:
            for name, param in backbone.named_parameters():
                if freeze_until in name:
                    logging.info(f"Parameters freezed until: {name}")
                    break

                logging.info(f"Freezing parameter: {name}")
                param.requires_grad = False

        elif freeze_until == 'all':
            for param in backbone.parameters():
                param.requires_grad = False

        elif freeze_until.lower() == 'none':
            logging.info("No parameters are frozen.")
            for param in backbone.parameters():
                param.requires_grad = True

        self.backbone = backbone
        self.projection_head = ProjectionHead(input_dim)
        self.classifier = nn.Linear(128, num_classes) if self.backbone_name in ['resnet18', 'densenet121'] else nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection_head(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # generate all backbones named parameters and save them as a text file
    for backbone_name in ['resnet18', 'densenet121', 'vit_small', 'vit_base', 'dinov2']:
        model = Backbone(backbone_name=backbone_name, pretrained=True, num_classes=10, freeze_until='all')
        print(f"Backbone: {backbone_name}")
        with open(f"{backbone_name}_named_parameters.txt", "w") as f:
            for name, param in model.named_parameters():
                f.write(f"{name}: {param.requires_grad}\n")
        print(f"Named parameters for {backbone_name} saved to {backbone_name}_named_parameters.txt")
        print("-" * 50)
