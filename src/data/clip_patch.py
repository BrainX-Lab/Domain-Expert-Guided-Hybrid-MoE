import os
import random
from typing import Optional, Tuple, Union, List
from functools import wraps
from random import randint

import torch
import torch.nn.functional as F

def _as_hw(patch_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Normalize patch size to (ph, pw)."""
    if isinstance(patch_size, int):
        return patch_size, patch_size
    if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
        ph, pw = int(patch_size[0]), int(patch_size[1])
        return ph, pw
    raise ValueError("patch_size must be int or tuple/list of length 2")

def random_padding(img: torch.Tensor, mask: torch.Tensor , target_size: Tuple[int, int] = (1024, 1024)):
    
    current_height, current_width = img.size()
    if current_width < target_size[1] or current_height < target_size[0]:
        pad_width = max(0, target_size[1] - current_width)
        pad_height = max(0, target_size[0] - current_height)
        
        pad_left = random.randint(0, pad_width)
        pad_top = random.randint(0, pad_height)
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top
        
        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    return img, mask

def random_mask_centered_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    patch_size: Union[int, Tuple[int, int]] = 1024,
    return_coords: bool = False,
):

    if mask is None:
        mask = image.bool().to(torch.uint8)
    H, W = mask.shape
    if image.shape[0] != H or image.shape[1] != W:
        raise ValueError("image and mask must have matching height and width")

    ph, pw = _as_hw(patch_size)

    if ph > H or pw > W:
        image, mask = random_padding(image, mask, target_size=(max(H, ph), max(W, pw)))
        H, W = mask.shape
    
    pos = torch.nonzero(mask.bool(), as_tuple=False)
    if pos.numel() == 0:
        raise ValueError("mask contains no positive pixels (no tumor region)")


    idx = randint(0, pos.shape[0]-1)
    py, px = pos[idx].tolist()

    y0_min = max(0, py - ph + 1)
    y0_max = min(py, H - ph)
    x0_min = max(0, px - pw + 1)
    x0_max = min(px, W - pw)

    if y0_min > y0_max or x0_min > x0_max:
        raise RuntimeError(f"No valid crop location")

    x0 = randint(x0_min, x0_max)
    y0 = randint(y0_min, y0_max)

    y1 = y0 + ph
    x1 = x0 + pw

    if image.ndim == 2:
        img_patch = image[y0:y1, x0:x1]
    else:
        img_patch = image[y0:y1, x0:x1, :]
    msk_patch = mask[y0:y1, x0:x1]

    if return_coords:
        return img_patch, msk_patch, (y0, x0, y1, x1)
    return img_patch, msk_patch


def clip_to_1024(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return random_mask_centered_crop(image, mask, patch_size=1024)


__all__ = [
    "random_mask_centered_crop",
    "clip_to_1024",
]
