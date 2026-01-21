"""
PyTorch Lightning DataModule for chest image classification.
"""

import lightning as L
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from ..data import get_dataset_class, create_data_loaders


class ChestDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for chest image classification."""
    
    def __init__(
        self,
        dataset_config: Dict[str, Any],
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ):
        """
        Initialize the DataModule.
        
        Args:
            dataset_config: Dataset configuration dictionary
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Get dataset class
        self.dataset_name = dataset_config.get('name', 'chest_xray')
        self.dataset_class = get_dataset_class(self.dataset_name)
        
        # Data loaders will be created in setup
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup data loaders for the given stage."""
        if stage == "fit" or stage is None:
            # Create train and validation loaders
            self.train_loader, self.val_loader, _ = create_data_loaders(
                self.dataset_config, self.dataset_class
            )
            print(f"Train loader: {len(self.train_loader)}")
            print(f"Val loader: {len(self.val_loader)}")
            print('', flush=True)
        
        if stage == "test" or stage is None:
            # Create test loader
            _, _, self.test_loader = create_data_loaders(
                self.dataset_config, self.dataset_class
            )
            print(f"Test loader: {len(self.test_loader)}")
            print('', flush=True)
    
    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        if self.train_loader is None:
            self.setup("fit")
        return self.train_loader
    
    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        # raise NotImplementedError("Validation data loader is not implemented")
        if self.val_loader is None:
            self.setup("fit")
        return self.val_loader
    
    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        if self.test_loader is None:
            self.setup("test")
        return self.test_loader
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after training/testing."""
        pass
