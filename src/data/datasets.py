"""
Dataset classes for chest image classification.
"""

import os
import csv
import random
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import pickle

from .clip_patch import random_mask_centered_crop
from .transforms import RandomRotation, RandomPadding, RandomElasticTransform
import matplotlib.pyplot as plt

class NewINbreastDataset(Dataset):
    """Dataset class for the updated INbreast dataset with CSV metadata and ROI masks."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split.lower()
        self.transform = transform
        self.gaze_transform = self._build_gaze_transform(transform)
        self.config = config or {}

        self.cross_val_config = (self.config.get("cross_validation") or {})
        self.cross_val_enabled = bool(self.cross_val_config.get("enabled", False))
        self.fold_column = self.cross_val_config.get("fold_column", "split")
        self.fold_prefix = self.cross_val_config.get("fold_prefix", "fold")
        self.current_fold = self.cross_val_config.get("current_fold")

        metadata_csv = self.cross_val_config.get("metadata_csv") if self.cross_val_enabled else None
        if not metadata_csv:
            metadata_csv = self.config.get("metadata_csv", "INbreast_long_corrected.csv")

        metadata_path = Path(metadata_csv)
        if not metadata_path.is_absolute():
            metadata_path = self.data_dir / metadata_path

        self.csv_path = metadata_path
        self.image_dir = self.data_dir / "im"
        self.mask_dir = self.data_dir / "ROI"
        gaze_dir_config = self.config.get("gaze_dir")
        # if gaze_dir_config is None:
        #     self.gaze_dir = self.data_dir / "hm_array"
        # else:
        gaze_dir_path = Path(gaze_dir_config)
        self.gaze_dir = gaze_dir_path if gaze_dir_path.is_absolute() else self.data_dir / gaze_dir_path
        if not self.gaze_dir.exists():
            raise FileNotFoundError(f"Gaze heatmap directory not found at {self.gaze_dir}")

        self.canonical_split = self._resolve_split_name(self.split)
        self.random_sampling = self.canonical_split == "train"
        self.config_epoch_length = int(self.config.get("epoch_length", 6000))

        patch_size = self.config.get("patch_size", self.config.get("image_size", [1024, 1024]))
        if isinstance(patch_size, (list, tuple)):
            if len(patch_size) == 2:
                self.patch_size: Tuple[int, int] = (int(patch_size[0]), int(patch_size[1]))
            else:
                size = int(patch_size[0])
                self.patch_size = (size, size)
        else:
            size = int(patch_size)
            self.patch_size = (size, size)

        # self.target_channels = self.config.get("channels", 1)

        self.samples, self.class_samples = self._load_samples()
        if self.random_sampling:
            self.available_labels = [label for label, items in self.class_samples.items() if items]
            if not self.available_labels:
                raise RuntimeError(
                    "NewINbreastDataset found no usable samples for any class; check CSV and filters."
                )
            self.epoch_length = self.config_epoch_length
        else:
            self.available_labels = []
            self.epoch_length = len(self.samples)
        
        # If validation/test, export each patch, save as png for showing
        # if self.split != "train":

        # self.export_patches()

    def _build_gaze_transform(self, transform: Optional[transforms.Compose]) -> Optional[transforms.Compose]:
        if transform is None or not isinstance(transform, transforms.Compose):
            return None

        allowed_ops = (
            transforms.Resize,
            transforms.RandomHorizontalFlip,
            transforms.RandomVerticalFlip,
            RandomRotation,
            RandomPadding,
            RandomElasticTransform,
        )

        gaze_ops: List[Any] = []
        for op in transform.transforms:
            if isinstance(op, allowed_ops):
                gaze_ops.append(copy.deepcopy(op))

        if not gaze_ops:
            return None

        return transforms.Compose(gaze_ops)

    def export_patches(self):
        export_dir = self.config.get("export_dir", 'data/new_INbreast/selected_patchs')+f'/{self.split}'
        if os.path.exists(export_dir): return
        os.makedirs(export_dir, exist_ok=True)
        for idx in range(len(self)):
            patch_tensor, gaze_patch, label, sid = self.getitem(idx, get_id=True)
            # print(patch_tensor.shape, label, flush=True)
            patch_np = patch_tensor.detach().cpu().numpy()
            patch_np = self._to_uint8(patch_np.mean(axis=0)) 
            # patch_image = Image.fromarray(patch_np, mode='L')
            
            plt.figure(figsize=(8,8))
            plt.imshow(patch_np, cmap='gray', vmin=0, vmax=255)
            gaze_np = gaze_patch.detach().cpu().squeeze(0).numpy()
            plt.imshow(gaze_np, cmap='jet', alpha=0.1, vmin=0, vmax=100)

            # patch_image = patch_image.convert("RGB")
            plt.savefig(f"{export_dir}/sid_{sid}_patch_{idx}.png")

    @staticmethod
    def _resolve_split_name(split: str) -> str:
        aliases = {
            "train": {"train", "training"},
            "val": {"val", "valid", "validation"},
            "test": {"test", "testing"},
        }
        normalized = split.lower()
        for canonical, names in aliases.items():
            if normalized == canonical or normalized in names:
                return canonical
        raise ValueError(f"Unsupported split '{split}' for NewINbreastDataset.")

    def _load_samples(self) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
        """Load usable samples from the CSV metadata grouped by class label."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found at {self.csv_path}")

        split_aliases = {
            "train": {"train", "training"},
            "val": {"val", "valid", "validation"},
            "test": {"test", "testing"},
        }
        valid_labels = {"0", "1", "2"}

        target_split = self.canonical_split

        target_fold_value = None
        if self.cross_val_enabled:
            target_fold_value = self._normalize_fold_identifier(self.current_fold)
            if target_fold_value is None:
                raise ValueError(
                    "Cross-validation enabled but no current_fold specified in configuration."
                )

        samples: List[Dict[str, Any]] = []
        class_samples: Dict[int, List[Dict[str, Any]]] = {0: [], 1: [], 2: []}
        
        with self.csv_path.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames or []
            if "YU Image Name" not in fieldnames:
                raise ValueError(
                    f"'YU Image Name' column not found in metadata CSV {self.csv_path}. "
                )
            for row in reader:
                usable = row.get("usable", "").strip().lower()
                if usable not in ("true", "1", "yes"):
                    continue

                if self.cross_val_enabled:
                    row_fold = row.get(self.fold_column, "").strip().lower()
                    if not row_fold:
                        continue
                    if self.canonical_split == "train":
                        if row_fold == target_fold_value:
                            continue
                    elif self.canonical_split in {"val", "test"}:
                        if row_fold != target_fold_value:
                            continue
                else:
                    row_split = row.get("split", "").strip().lower()
                    resolved_split = None
                    for key, aliases in split_aliases.items():
                        if row_split in aliases or row_split == key:
                            resolved_split = key
                            break
                    if resolved_split != target_split:
                        continue

                label_str = row.get("new_label", "").strip()
                if label_str not in valid_labels:
                    continue

                file_id = row.get("File Name", "").strip()
                if not file_id:
                    continue

                image_path = self.image_dir / f"{file_id}.tiff"
                if not image_path.exists():
                    continue

                gaze_name = row.get("YU Image Name", "").strip()
                if not gaze_name:
                    continue
                gaze_path = self.gaze_dir / f"{gaze_name}.npy"
                if not gaze_path.exists():
                    continue

                label = int(label_str)
                mask_path = None
                if label != 0:
                    candidate_mask = self.mask_dir / f"{file_id}_mask.tiff"
                    if candidate_mask.exists():
                        mask_path = candidate_mask

                sample = {
                    "file_id": file_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "label": label,
                    "gaze_path": gaze_path,
                }
                samples.append(sample)
                class_samples[label].append(sample)

        if not samples:
            raise RuntimeError(
                f"No samples found for split '{self.split}' in NewINbreastDataset. "
                "Check CSV metadata and configuration."
            )

        return samples, class_samples, 

    def _normalize_fold_identifier(self, fold_value: Optional[Any]) -> Optional[str]:
        """Normalize different fold representations into a canonical lowercase form."""
        if fold_value is None:
            return None

        if isinstance(fold_value, int):
            return f"{self.fold_prefix}{fold_value}".lower()

        fold_str = str(fold_value).strip().lower()
        if not fold_str:
            return None

        if self.fold_prefix and not fold_str.startswith(self.fold_prefix):
            return f"{self.fold_prefix}{fold_str}".lower()

        return fold_str

    @staticmethod
    def _to_uint8(array: np.ndarray) -> np.ndarray:
        """Convert any numeric array to uint8 for PIL compatibility."""
        if array.dtype == np.uint8:
            return array
        array = array.astype(np.float32)
        array = array - array.min()
        max_val = array.max()
        if max_val > 0:
            array = array / max_val
        array = (array * 255.0).clip(0, 255)
        return array.astype(np.uint8)

    def __len__(self) -> int:
        if self.random_sampling:
            return self.epoch_length
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.getitem(idx, False)

    def getitem(self, idx: int, get_id: bool = False) -> Tuple[torch.Tensor, torch.Tensor, int]:

        if self.random_sampling:
            chosen_label = random.choice(self.available_labels)
            sample = random.choice(self.class_samples[chosen_label])
        else:
            sample = self.samples[idx]
        image_path: Path = sample["image_path"]
        mask_path: Optional[Path] = sample["mask_path"]
        label: int = sample["label"]
        gaze_path: Path = sample["gaze_path"]

        with Image.open(image_path) as img:
            image_array = np.array(img, copy=True)

        image_tensor = torch.from_numpy(image_array)
        image_tensor = image_tensor.to(torch.float32) / np.iinfo(image_array.dtype).max  # normalize to [0, 1]

        gaze_array = np.load(gaze_path)
        gaze_tensor_full = torch.from_numpy(gaze_array).to(torch.float32)
        if gaze_tensor_full.ndim == 3:
            gaze_tensor_full = gaze_tensor_full[..., 0]

        mask_tensor: Optional[torch.Tensor] = None
        if label != 0 and mask_path is not None:
            with Image.open(mask_path) as mask_img:
                mask_array = np.array(mask_img, copy=True)
            if mask_array.ndim == 3:
                mask_array = mask_array[..., 0]
            mask_tensor = torch.from_numpy(mask_array).to(torch.uint8)
            mask_tensor = (mask_tensor > 0).to(torch.uint8)

        if self.split == "train":  # random scaled choose if training:
            scale_rate = random.uniform(0.5, 1.5)
            patch_size_scaled = (int(self.patch_size[0] * scale_rate), int(self.patch_size[1] * scale_rate))
        else:
            patch_size_scaled = self.patch_size

        patch_tensor, _, coords = random_mask_centered_crop(
            image_tensor,
            mask_tensor if mask_tensor is not None else None,
            patch_size=patch_size_scaled,
            return_coords=True,
        )

        y0, x0, y1, x1 = coords
        gaze_patch = gaze_tensor_full[y0:y1, x0:x1]

        if gaze_patch.numel() == 0:
            raise ValueError(f"Gaze patch extracted from {gaze_path} is empty with coords {coords}.")

        if patch_tensor.shape != patch_size_scaled:
            print(f"Warning: patch_tensor shape {patch_tensor.shape} does not match expected {patch_size_scaled}", flush=True)
            raise ValueError("Patch extraction failed to produce expected size.")

        patch_tensor = patch_tensor.to(torch.float32)
        gaze_patch = gaze_patch.to(torch.float32)

        patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0)
        gaze_patch = gaze_patch.unsqueeze(0).unsqueeze(0)

        rand_state_before = random.getstate()
        torch_state_before = torch.random.get_rng_state()

        if self.transform:
            patch_tensor = self.transform(patch_tensor)
        patch_tensor = patch_tensor[0]

        rand_state_after = random.getstate()
        torch_state_after = torch.random.get_rng_state()

        if self.gaze_transform:
            random.setstate(rand_state_before)
            torch.random.set_rng_state(torch_state_before)
            gaze_patch = self.gaze_transform(gaze_patch)
            random.setstate(rand_state_after)
            torch.random.set_rng_state(torch_state_after)
        gaze_patch = gaze_patch[0]

        if patch_tensor.shape != (3, 1024, 1024):
            print(f"Warning: patch_tensor shape {patch_tensor.shape} is not as expected (3, 1024, 1024)", flush=True)
            raise ValueError("Transformation resulted in unexpected tensor shape.")

        if gaze_patch.shape[0] != 1 or gaze_patch.shape[1] != 1024 or gaze_patch.shape[2] != 1024:
            print(f"Warning: gaze_patch shape {gaze_patch.shape} is not as expected (1, 1024, 1024)", flush=True)
            raise ValueError("Gaze transformation resulted in unexpected tensor shape.")

        if get_id:
            return patch_tensor, gaze_patch, label, sample["file_id"]
        return patch_tensor, gaze_patch, label


def get_dataset_class(dataset_name: str):
    dataset_classes = {
        'new_inbreast': NewINbreastDataset
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_classes.keys())}")
    
    return NewINbreastDataset
