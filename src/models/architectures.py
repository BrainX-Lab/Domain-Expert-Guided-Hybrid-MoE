"""
Model architectures for chest image classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any, List
import timm
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152

class ResNetClassifier(nn.Module):
    """ResNet-based classifier for chest image classification."""
    
    def __init__(self, model_name: str = 'resnet50', 
                 num_classes: int = 2,
                 input_channels: int = 3,
                 freeze_backbone: bool = False,
                 dropout_rate: float = 0.5,
                 use_moe: bool = False,
                 moe_num_experts: int = 4,
                 moe_top_k: int = 2,
                #  moe_load_balance_weight: float = 0.01,
                 moe_router_type: str = 'image',
                 moe_distillation_active: bool = False,
                 pretrained: bool = False,
                 **kwargs):
        super(ResNetClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.freeze_backbone = freeze_backbone
        self.use_moe = use_moe
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_router_type = moe_router_type
        self.moe_distillation_active = moe_distillation_active
        
        resnet_functions = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152,
        }
        
        if model_name not in resnet_functions:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        resnet_kwargs = {
            'num_classes': num_classes,
            'input_channels': input_channels,
            'pretrained': pretrained,
            **kwargs
        }
        
        if use_moe:
            resnet_kwargs.update({
                'use_moe': use_moe,
                'moe_num_experts': moe_num_experts,
                'moe_top_k': moe_top_k,
                'moe_router_type': moe_router_type,
                'moe_distillation_active': kwargs.get('moe_distillation_active', moe_distillation_active),
            })
        
        self.backbone = resnet_functions[model_name](**resnet_kwargs)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x, expert_feature: Optional[torch.Tensor] = None):
        """Forward pass."""
        return self.backbone(x, expert_feature)
    
    def get_moe_load_balance_loss(self):
        """Get the total load balancing loss from all MoE layers in the model."""
        if not self.use_moe:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.backbone.get_moe_load_balance_loss()


def create_model(config: Dict[str, Any]) -> nn.Module:
    model_name = config.get('name', 'resnet50')
    num_classes = config.get('num_classes', 2)
    input_channels = config.get('input_channels', 3)
    freeze_backbone = config.get('freeze_backbone', False)
    dropout_rate = config.get('dropout_rate', 0.5)
    pretrained = config.get('pretrained', False)
    
    if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        moe_config = config.get('moe_config', {})
        use_moe = moe_config.get('use_moe', False)
        
        if use_moe:
            return ResNetClassifier(
                model_name=model_name,
                num_classes=num_classes,
                input_channels=input_channels,
                freeze_backbone=freeze_backbone,
                dropout_rate=dropout_rate,
                use_moe=True,
                moe_num_experts=moe_config.get('num_of_expert', 4),
                moe_top_k=moe_config.get('top_k', 2),
                moe_router_type=moe_config.get('moe_router_type', 'both'),
                moe_distillation_active=moe_config.get('distillation_active', False),
                pretrained=pretrained
            )
        else:
            return ResNetClassifier(
                model_name=model_name,
                num_classes=num_classes,
                input_channels=input_channels,
                freeze_backbone=freeze_backbone,
                dropout_rate=dropout_rate,
                pretrained=pretrained
            )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

class ExpertModelEyegaze(nn.Module):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = config or {}

        pretrained = config.get('pretrained', False)
        in_chans = config.get('in_chans', config.get('input_channels', 1))
        output_dim = config.get('output_dim', 128)

        self.backbone = timm.create_model(
            'resnet18',
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
        )
        feature_dim = getattr(self.backbone, 'num_features', None)
        if feature_dim is None:
            raise AttributeError("The timm resnet18 backbone must expose num_features.")

        if config.get('freeze_backbone', False):
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Linear(feature_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)

        if isinstance(features, (list, tuple)):
            features = features[0]

        if isinstance(features, torch.Tensor) and features.dim() > 2:
            features = torch.flatten(features, 1)

        return self.projection(features)

def create_expert_model(config):
    if config is None:
        config = {}

    return ExpertModelEyegaze(config)

class ExpGuidedModel(nn.Module):

    def __init__(self, config: Dict[str, Any], expert_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model = create_model(config)

        resolved_expert_config = expert_config
        if resolved_expert_config is None:
            resolved_expert_config = config.get('expert_model') or config.get('expert_config')

        self.expert_model = create_expert_model(resolved_expert_config)

    def forward(self, x: torch.Tensor, expert_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if expert_x is not None:
            expert_feature = self.expert_model(expert_x)
        else:
            expert_feature = None
        return self.model(x, expert_feature)

def create_exp_guided_model(config, expert_config=None):
    return ExpGuidedModel(config, expert_config)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
