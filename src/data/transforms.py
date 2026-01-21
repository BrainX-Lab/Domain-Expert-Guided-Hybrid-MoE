"""
Data transformation utilities for chest image classification.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import random


class RandomRotation:
    """Random rotation with padding to avoid black borders."""
    
    def __init__(self, degrees: float = 15, p: float = 0.5, fill_mode: str = 'reflect'):
        self.degrees = degrees
        self.p = p
        self.fill_mode = fill_mode  # 'reflect', 'constant', 'edge', 'symmetric'
    
    def __call__(self, img):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            # Use reflect padding to avoid black borders
            return F.rotate(img, angle, fill=0, interpolation=F.InterpolationMode.BILINEAR)
        return img

class RandomBrightnessContrast:
    """Random brightness and contrast adjustment."""
    
    def __init__(self, brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.5):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            # Brightness adjustment
            brightness_factor = random.uniform(*self.brightness_range)
            img = F.adjust_brightness(img, brightness_factor)
            
            # Contrast adjustment
            contrast_factor = random.uniform(*self.contrast_range)
            img = F.adjust_contrast(img, contrast_factor)
        
        return img


class RandomPadding:
    """Random padding to reach target size with zeros."""
    
    def __init__(self, target_size: Tuple[int, int] = (1024, 1024), p: float = 1.0):
        """
        Initialize random padding transform.
        
        Args:
            target_size: Target size (width, height) to pad to
            p: Probability of applying padding (default 1.0 for always apply)
        """
        self.target_size = target_size
        self.p = p
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape)==4:
            _b_c=True
            img=img[0,0,...]
        else:
            _b_c=False
            
        if random.random() < self.p:
            # Get current image size
            current_width, current_height = img.size()
            
            # Check if image is smaller than target size
            if current_width < self.target_size[0] or current_height < self.target_size[1]:
                # Calculate padding needed
                pad_width = max(0, self.target_size[0] - current_width)
                pad_height = max(0, self.target_size[1] - current_height)
                
                # Calculate random padding positions
                pad_left = random.randint(0, pad_width)
                pad_top = random.randint(0, pad_height)
                pad_right = pad_width - pad_left
                pad_bottom = pad_height - pad_top
                
                # Apply padding with zeros (black padding)
                if img.mode == 'L':
                    # Grayscale image
                    img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
                else:
                    # RGB image
                    img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0))
        if _b_c:
            img=img.reshape(1,1,*img.shape)
        return img


# class SimpleNormalize:
#     """Simple normalization from 0-255 to 0-1 range."""
    
#     def __init__(self, max_value: float = 255.0):
#         """Initialize simple normalization (divide by 255)."""
#         self.max_value = max_value
    
#     def __call__(self, tensor):
#         """Normalize tensor from 0-255 range to 0-1 range."""
#         return tensor / self.max_value


class RandomNoise:
    """Add random Gaussian noise to image."""
    
    def __init__(self, noise_factor: float = 0.1, p: float = 0.3):
        self.noise_factor = noise_factor
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            noise=torch.randn_like(img) * self.noise_factor
            img = img + noise
            img = torch.clamp(img, 0.0, 1.0)
        return img


class RandomElasticTransform:
    """Apply elastic transformation with a given probability."""
    
    def __init__(self, alpha: float = 50.0, sigma: float = 5.0, p: float = 0.5):
        self.transform = transforms.ElasticTransform(alpha=alpha, sigma=sigma)
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            return self.transform(img)
        return img


class CopyToThreeChannels:
    """Copy single channel image to three channels."""
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # if len(img.shape) == 2:
        #     img = img.unsqueeze(0)
        if img.shape[1] == 1:
            return img.repeat(1, 3, 1, 1)
        return img

def get_train_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    Training transforms: 6 steps (padding → rotation → color aug → noise → tensor → normalize)
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Composed transforms for training
    """
    image_size = config.get('image_size', [224, 224])
    augmentation_config = config.get('augmentation', {})
    dataset_name = config.get('name', '')
    
    transform_list = []
    
    # For INbreast dataset, use original image size without resize/crop
    if dataset_name =='inbreast':
    
        # No resize or crop needed - use original 1024x1024 images directly
        # Add padding for images smaller than 1024x1024
        transform_list.append(RandomPadding(target_size=(1024, 1024), p=1.0))
    elif dataset_name=='new_inbreast':
        transform_list.append(transforms.Resize(image_size))

    if dataset_name in ('inbreast', 'new_inbreast'):
        # Step 2: Apply geometric augmentations first (on larger image)
        # Random horizontal flip (disabled for mammography)
        if augmentation_config.get('horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        # Random vertical flip
        if augmentation_config.get('vertical_flip', False):
            transform_list.append(transforms.RandomVerticalFlip(p=0.5))

        # Random rotation
        if augmentation_config.get('rotation', 0) > 0:
            transform_list.append(RandomRotation(degrees=augmentation_config['rotation'], p=0.5))

        # Elastic transform
        if augmentation_config.get('elastic_transform', False):
            alpha = augmentation_config.get('elastic_alpha', 50.0)
            sigma = augmentation_config.get('elastic_sigma', 5.0)
            p = augmentation_config.get('elastic_p', 0.5)
            transform_list.append(RandomElasticTransform(alpha=alpha, sigma=sigma, p=p))
        
        # Step 3: Apply color augmentations (on original image)
        # Brightness and contrast
        if augmentation_config.get('brightness_range') or augmentation_config.get('contrast_range'):
            brightness_range = augmentation_config.get('brightness_range', (0.8, 1.2))
            contrast_range = augmentation_config.get('contrast_range', (0.8, 1.2))
            transform_list.append(RandomBrightnessContrast(
                brightness_range=brightness_range,
                contrast_range=contrast_range,
                p=0.5
            ))
        
        # Random noise
        if augmentation_config.get('noise_factor', 0) > 0:
            transform_list.append(RandomNoise(noise_factor=augmentation_config.get('noise_factor', 0.05), p=0.3))
        
        channels = config.get('channels', 3)
        if channels == 1:
            transform_list.append(CopyToThreeChannels())

        # RGB normalization
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
    
    return transforms.Compose(transform_list)


def get_val_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    Get validation data transformations.
    Validation transforms: 3 steps (padding → tensor → normalize)

    Args:
        config: Configuration dictionary
        
    Returns:
        Composed transforms for validation
    """
    image_size = config.get('image_size', [224, 224])
    dataset_name = config.get('name', '')
    
    transform_list = []
    
    # For INbreast dataset, use original image size without resize
    if dataset_name == 'inbreast':
        # No resize needed - use original 1024x1024 images directly
        # Add padding for images smaller than 1024x1024
        transform_list.append(RandomPadding(target_size=(1024, 1024), p=1.0))
    elif dataset_name=='new_inbreast':
        transform_list.append(transforms.Resize(image_size))

    else:
        # Standard resize for other datasets
        transform_list.append(transforms.Resize(image_size))
    
    # Common transforms for all datasets
    # transform_list.append(transforms.ToTensor())
    
    # normalization
    channels = config.get('channels', 3)
    if channels == 1:
        transform_list.append(CopyToThreeChannels())

    # RGB normalization
    transform_list.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    
    return transforms.Compose(transform_list)


def get_test_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    Get test data transformations.
    Test transforms: 3 steps (resize → tensor → normalize)

    Args:
        config: Configuration dictionary
        
    Returns:
        Composed transforms for test
    """
    return get_val_transforms(config)


def create_data_loaders(config: Dict[str, Any], dataset_class) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        config: Configuration dictionary
        dataset_class: Dataset class to use
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = config['data_dir']
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    
    # Get transforms
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)
    test_transforms = get_test_transforms(config)
    
    # Create datasets
    train_dataset = dataset_class(
        data_dir=data_dir,
        split='train',
        transform=train_transforms,
        config=config
    )
    
    val_dataset = dataset_class(
        data_dir=data_dir,
        split='val',
        transform=val_transforms,
        config=config
    )
    
    test_dataset = dataset_class(
        data_dir=data_dir,
        split='test',
        transform=test_transforms,
        config=config
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
