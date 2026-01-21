"""
Pretrained weights utilities for VGG and ResNet models.

This module provides functionality to download and load ImageNet pretrained weights,
with special handling for MoE (Mixture of Experts) models where weights need to be
copied to each expert.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import torchvision.models as models
from torchvision.models import VGG, ResNet
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


def download_imagenet_weights(model_name: str, progress: bool = True) -> Dict[str, torch.Tensor]:
    """
    Download ImageNet pretrained weights for a given model.
    """
    # Map model names to torchvision functions
    model_functions = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }
    
    if model_name not in model_functions:
        raise ValueError(f"Unsupported model: {model_name}. "
                        f"Supported models: {list(model_functions.keys())}")
    
    logger.info(f"Downloading ImageNet pretrained weights for {model_name}...")
    
    # Load pretrained model
    pretrained_model = model_functions[model_name](pretrained=True, progress=progress)
    
    # Extract state dict
    state_dict = pretrained_model.state_dict()
    
    logger.info(f"Successfully downloaded pretrained weights for {model_name}")
    return state_dict


def adapt_weights_for_channels(state_dict: Dict[str, torch.Tensor], 
                              target_model: nn.Module,
                              input_channels: int = 3) -> Dict[str, torch.Tensor]:
    """
    Adapt pretrained weights for different input channel configurations.
    
    Args:
        state_dict: Pretrained state dictionary
        target_model: Target model to adapt weights for
        input_channels: Number of input channels in target model
        
    Returns:
        Adapted state dictionary
    """
    adapted_state_dict = state_dict.copy()
    
    first_conv_keys = ['features.0.weight', 'conv1.weight']  # first conv keys
    
    for key in first_conv_keys:
        if key in state_dict:
            original_weight = state_dict[key]
            original_channels = original_weight.shape[1]
            
            if original_channels != input_channels:
                logger.info(f"Adapting {key} from {original_channels} to {input_channels} channels")
                
                if input_channels == 1 and original_channels == 3:
                    adapted_weight = original_weight.mean(dim=1, keepdim=True)
                elif input_channels == 3 and original_channels == 1:
                    adapted_weight = original_weight.repeat(1, 3, 1, 1)
                else:
                    adapted_weight = original_weight[:, :input_channels, :, :]
                
                adapted_state_dict[key] = adapted_weight
    
    return adapted_state_dict


def copy_weights_to_moe_experts(state_dict: Dict[str, torch.Tensor], 
                               target_model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Copy weights to MoE experts when using MoE layers.
    """
    adapted_state_dict = state_dict.copy()
    
    # Find MoE layers in the target model
    moe_layers = {}
    for name, module in target_model.named_modules():
        if hasattr(module, 'experts') and isinstance(module.experts, nn.ModuleList):
            moe_layers[name] = module
    
    # Copy weights to MoE experts
    for moe_name, moe_layer in moe_layers.items():
        logger.info(f"Copying weights to MoE layer: {moe_name}")
        moe_name_for_search=moe_name
        if moe_name_for_search.find("moelayer_route_from")!=-1:
            moe_name_for_search=moe_name_for_search[:moe_name_for_search.rfind('.')]
        
        for expert_idx, expert in enumerate(moe_layer.experts):
            expert_state_dict = expert.state_dict()
            
            # Copy weights from pretrained model to each expert
            for param_name, param_value in expert_state_dict.items():
                
                pretrained_key = moe_name_for_search + f".{param_name}"
                
                # # Simple heuristic: look for similar parameter names
                # for pretrained_key_candidate in state_dict.keys():
                #     if param_name in pretrained_key_candidate:
                #         pretrained_key = pretrained_key_candidate
                #         break
                if pretrained_key and pretrained_key in state_dict:
                    expert_key = f"{moe_name}.experts.{expert_idx}.{param_name}"
                    adapted_state_dict[expert_key] = state_dict[pretrained_key].clone()
                    logger.debug(f"Copied {pretrained_key} to {expert_key}")
                else:
                    logger.warning(f"No matching pretrained key found for {param_name} in MoE layer {moe_name}")
    
    return adapted_state_dict


def load_pretrained_weights(model: nn.Module, 
                           model_name: str,
                           input_channels: int = 3,
                           use_moe: bool = False,
                           strict: bool = False,
                           progress: bool = True,
                           exclude_classifier: bool = True) -> nn.Module:
    
    logger.info(f"Loading pretrained weights for {model_name}")
    
    # Download pretrained weights
    pretrained_state_dict = download_imagenet_weights(model_name, progress=progress)
    
    # Adapt weights for different input channels
    if input_channels != 3:
        pretrained_state_dict = adapt_weights_for_channels(
            pretrained_state_dict, model, input_channels
        )
    
    # Copy weights to MoE experts if needed
    if use_moe:
        pretrained_state_dict = copy_weights_to_moe_experts(
            pretrained_state_dict, model
        )
    
    # Exclude classifier layers if requested
    if exclude_classifier:
        classifier_keys_to_remove = []
        for key in pretrained_state_dict.keys():
            if any(classifier_keyword in key.lower() for classifier_keyword in 
                   ['classifier', 'fc', 'head', 'heads']):
                classifier_keys_to_remove.append(key)
        
        for key in classifier_keys_to_remove:
            del pretrained_state_dict[key]
            logger.info(f"Excluded classifier layer: {key}")
    try:
        model.load_state_dict(pretrained_state_dict, strict=strict)
        logger.info(f"Successfully loaded pretrained weights for {model_name}")
    except Exception as e:
        logger.warning(f"Failed to load some weights: {e}")
        if not strict:
            model.load_state_dict(pretrained_state_dict, strict=False)
            logger.info("Loaded pretrained weights with strict=False")
        else:
            raise e
    
    return model


def load_resnet_pretrained(model: nn.Module, 
                          model_name: str,
                          input_channels: int = 3,
                          use_moe: bool = False,
                          **kwargs) -> nn.Module:
    """Load pretrained weights for ResNet models."""
    return load_pretrained_weights(
        model, model_name, input_channels, use_moe, **kwargs
    )
