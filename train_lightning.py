#!/usr/bin/env python3
"""
Main training script using PyTorch Lightning for chest image classification.
"""

import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import os
import sys
from pathlib import Path

from src.utils.config import load_config
from src.lightning import ChestImageClassifier, ChestDataModule


def get_device_config(config):
    """Get device configuration for Lightning with robust error handling."""
    device_config = config.get('device', 'auto')
    
    if device_config == 'auto':
        try:
            cuda_devices = L.pytorch.accelerators.find_usable_cuda_devices()
            if cuda_devices:
                return "cuda"
        except Exception:
            pass
        return "cpu"
    else:
        valid_devices = ['cpu', 'cuda', 'auto']
        if device_config not in valid_devices:
            print(f"Warning: Invalid device '{device_config}' specified. Falling back to 'cpu'.")
            return "cpu"
        return device_config


def create_callbacks(config, log_dir):
    """Create Lightning callbacks."""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_config = config.get('training', {}).get('checkpoint', {})
    if checkpoint_config.get('save_best', True):
        checkpoint_callback = ModelCheckpoint(
            dirpath=log_dir / "checkpoints",
            filename="best-epoch_{epoch:02d}-valacc_{val/accuracy:.4f}",
            auto_insert_metric_name=False,
            monitor="val/accuracy",
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping_config = config.get('training', {}).get('early_stopping', {})
    if early_stopping_config.get('enabled', True):
        early_stopping_callback = EarlyStopping(
            monitor="val/accuracy",
            mode="max",
            patience=early_stopping_config.get('patience', 15),
            min_delta=early_stopping_config.get('min_delta', 0.001),
            verbose=True
        )
        callbacks.append(early_stopping_callback)
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks


def create_logger(config, experiment_name, log_dir=None):
    """Create Lightning logger."""
    logging_config = config.get('logging', {})
    
    if log_dir is None:
        log_dir = Path(logging_config.get('log_dir', 'experiments/logs'))
    return TensorBoardLogger(
        save_dir=str(log_dir),
        name=experiment_name,
        version=None
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train chest image classification model with PyTorch Lightning')
    parser.add_argument('--config', type=str, default='configs/exmoe_exp3_crossval.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Override data directory from config')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Override model name from config')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs from config')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Override learning rate from config')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--test_only', action='store_true',
                       help='Only run testing (no training)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.data_dir:
        config.set('dataset.data_dir', args.data_dir)
    if args.model_name:
        config.set('model.name', args.model_name)
    if args.epochs:
        config.set('training.epochs', args.epochs)
    if args.batch_size:
        config.set('dataset.batch_size', args.batch_size)
    if args.learning_rate:
        config.set('training.learning_rate', args.learning_rate)
    
    if not config.validate():
        print("Configuration validation failed!")
        return
    
    device = get_device_config(config)
    print(f"Using device: {device}")
    
    experiment_name = args.experiment_name or config.get('exp_name') or f"{config.get('model.name')}_{config.get('dataset.name')}"
    log_dir = Path(f"experiments/{experiment_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    config.save(log_dir / "config.yaml")
    
    print("Setting up data module...")
    data_module = ChestDataModule(
        dataset_config=config.get('dataset'),
        batch_size=config.get('dataset.batch_size', 32),
        num_workers=config.get('dataset.num_workers', 4)
    )
    
    print("Creating model...")
    model = ChestImageClassifier(
        model_config=config.get('model'),
        training_config=config.get('training'),
        num_classes=config.get('model.num_classes', 2),
        learning_rate=config.get('training.learning_rate', 1e-3),
        weight_decay=config.get('training.weight_decay', 1e-4),
        scheduler_type=config.get('training.scheduler', 'cosine'),
        dataset_config=config.get('dataset')
    )
    
    print(f"Model: {config.get('model.name')}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    callbacks = create_callbacks(config, log_dir)
    
    logger = create_logger(config, experiment_name, log_dir)
    
    trainer = L.Trainer(
        accelerator=device,
        devices=1,
        max_epochs=config.get('training.epochs', 100),
        callbacks=callbacks,
        logger=logger,
        precision="16-mixed" if config.get('mixed_precision', True) else 32,
        # precision=32,
        gradient_clip_val=config.get('training.gradient_clip_val', None),
        accumulate_grad_batches=config.get('gradient_accumulation_steps', 1),
        log_every_n_steps=50,
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    if not args.test_only:
        print("Starting training...")
        trainer.fit(
            model, 
            data_module,
            ckpt_path=args.resume_from_checkpoint
        )
    
    # Test model
    print("Testing model...")
    trainer.test(model, data_module, ckpt_path="best" if not args.test_only else args.resume_from_checkpoint)
    
    print(f"\nTraining completed! Results saved to {log_dir}")


if __name__ == "__main__":
    main()
