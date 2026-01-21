"""
PyTorch Lightning module for chest image classification.
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
from pathlib import Path
import time

from ..models.architectures import create_exp_guided_model
from ..models.moe import MoELayer, MoelayerRouteFromBoth


class ChestImageClassifier(L.LightningModule):
    """PyTorch Lightning module for chest image classification."""
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        num_classes: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_type: str = "plateau",
        dataset_config: Dict[str, Any]=None,
        **kwargs
    ):
        """
        Initialize the Lightning module.
        
        Args:
            model_config: Model configuration dictionary
            training_config: Training configuration dictionary
            num_classes: Number of output classes
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            scheduler_type: Type of learning rate scheduler
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model_config = model_config
        self.training_config = training_config
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.dataset_config = dataset_config
        
        # Create model
        self.model = create_exp_guided_model(model_config)
        
        # Check if model has MoE layers
        self.has_moe = self._detect_moe_layers()
        self.moe_config = self._extract_moe_config(model_config)
        # Track whether distillation checkpoint has been loaded
        self._distill_ckpt_loaded = False
        # Try to initialize from checkpoint now (will try again at fit start for cross-val paths)
        self._init_distillation_from_ckpt(model_config)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_accuracy_per_class = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.val_accuracy_per_class = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.test_accuracy_per_class = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        
        self.train_precision = Precision(task="multiclass", num_classes=num_classes, average="weighted")
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="weighted")
        self.train_precision_per_class = Precision(task="multiclass", num_classes=num_classes, average=None)
        self.val_precision_per_class = Precision(task="multiclass", num_classes=num_classes, average=None)
        self.test_precision_per_class = Precision(task="multiclass", num_classes=num_classes, average=None)
        self.val_precision_no_gaze = Precision(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_precision_no_gaze = Precision(task="multiclass", num_classes=num_classes, average="weighted")
        
        self.train_recall = Recall(task="multiclass", num_classes=num_classes, average="weighted")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="weighted")
        self.train_recall_per_class = Recall(task="multiclass", num_classes=num_classes, average=None)
        self.val_recall_per_class = Recall(task="multiclass", num_classes=num_classes, average=None)
        self.test_recall_per_class = Recall(task="multiclass", num_classes=num_classes, average=None)
        self.val_recall_no_gaze = Recall(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_recall_no_gaze = Recall(task="multiclass", num_classes=num_classes, average="weighted")
        
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.train_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)
        self.val_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)
        self.test_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)
        self.val_f1_no_gaze = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_f1_no_gaze = F1Score(task="multiclass", num_classes=num_classes, average="weighted")

        self.val_accuracy_no_gaze = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy_no_gaze = Accuracy(task="multiclass", num_classes=num_classes)
        
        # AUC metrics
        self.train_auc_per_class = None
        self.val_auc_per_class = None
        self.test_auc_per_class = None

        # Control cadence for histogram logging of MoE router outputs
        self.moe_hist_every_n_steps = training_config.get("moe_hist_every_n_steps", 200)
        
        if num_classes == 2:
            self.train_auc = AUROC(task="binary")
            self.val_auc = AUROC(task="binary")
            self.test_auc = AUROC(task="binary")
            self.val_auc_no_gaze = AUROC(task="binary")
            self.test_auc_no_gaze = AUROC(task="binary")
        else:
            self.train_auc = AUROC(task="multiclass", num_classes=num_classes)
            self.val_auc = AUROC(task="multiclass", num_classes=num_classes)
            self.test_auc = AUROC(task="multiclass", num_classes=num_classes)
            self.val_auc_no_gaze = AUROC(task="multiclass", num_classes=num_classes)
            self.test_auc_no_gaze = AUROC(task="multiclass", num_classes=num_classes)
            self.train_auc_per_class = AUROC(
                task="multiclass", num_classes=num_classes, average=None
            )
            self.val_auc_per_class = AUROC(
                task="multiclass", num_classes=num_classes, average=None
            )
            self.test_auc_per_class = AUROC(
                task="multiclass", num_classes=num_classes, average=None
            )
        
        # Store predictions for final evaluation
        self.test_predictions = []
        self.test_targets = []
        self.test_probabilities = []
        
        # Progress bar throttling (30 seconds)
        self.last_prog_bar_update = 0.0
        self.prog_bar_throttle_interval = 30.0  # seconds
    
    def _detect_moe_layers(self) -> bool:
        """Detect if the model contains MoE layers."""
        for module in self.model.modules():
            if isinstance(module, MoELayer):
                return True
        return False
    
    def _extract_moe_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract MoE configuration from model config."""
        moe_config = model_config.get('moe_config', {})
        return {
            'load_balance_weight': moe_config.get('load_balance_weight', 0.01),
            'distillation_weight': moe_config.get('distillation_weight', 0.1),
            'distillation_active': moe_config.get('distillation_active', False),
            'distillation_checkpoint': moe_config.get('distillation_checkpoint', None),
            'num_of_expert': moe_config.get('num_of_expert', 4),
            'top_k': moe_config.get('top_k', 2),
        }

    def _init_distillation_from_ckpt(self, model_config: Dict[str, Any]) -> None:
        """Load a checkpoint to initialize model parameters.

        If cross-validation is enabled and the current fold is known via the
        attached DataModule, load from `<ckpt_path>/<current_fold>/checkpoints/last.ckpt`.
        Otherwise, load directly from `ckpt_path`.
        """
        if self._distill_ckpt_loaded:
            return
        moe_cfg = (model_config or {}).get('moe_config', {})
        # if not moe_cfg.get('distillation_active', False):
        #     return
        ckpt_path = moe_cfg.get('distillation_checkpoint')
        if not ckpt_path:
            return
        # Resolve cross-validation-aware path if possible
        ckpt_file = Path(ckpt_path)
        current_fold = None
    
        ds_cfg = self.dataset_config
        cv_cfg = ds_cfg.get('cross_validation') or {}
        if cv_cfg.get('enabled') and cv_cfg.get('current_fold'):
            current_fold = str(cv_cfg.get('current_fold'))        
            fold_ckpt = Path(ckpt_path) / current_fold / 'checkpoints' / 'last.ckpt'
            ckpt_file = fold_ckpt
        else:
            ckpt_file = Path(ckpt_path)

        if not ckpt_file.exists():
            print(f"[Distill] Checkpoint not found: {ckpt_file}")
            raise Exception()
            return

        ckpt = torch.load(str(ckpt_file), weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)

        # Prefer keys under 'model.' (LightningModule scope). Strip prefix for self.model
        prefixed = {k: v for k, v in state_dict.items() if k.startswith('model.')}
        if prefixed:
            trimmed = {k[len('model.'):]: v for k, v in prefixed.items()}
            msg_missing, msg_unexpected = self.model.load_state_dict(trimmed, strict=False)
            print(f"[Distill] Loaded from {ckpt_file} with {len(msg_missing)} missing and {len(msg_unexpected)} unexpected keys.")
            self._distill_ckpt_loaded = True
            return

        # Fallback: intersect by exact keys of self.model
        target_sd = self.model.state_dict()
        intersect = {k: v for k, v in state_dict.items() if k in target_sd and getattr(v, 'shape', None) == target_sd[k].shape}
        if intersect:
            msg_missing, msg_unexpected = self.model.load_state_dict(intersect, strict=False)
            print(f"[Distill] Partially loaded from {ckpt_file}: {len(intersect)} tensors, {len(msg_missing)} missing, {len(msg_unexpected)} unexpected.")
            self._distill_ckpt_loaded = True
        else:
            print(f"[Distill] No matching parameters found in checkpoint: {ckpt_file}")
        # except Exception as e:
        #     print(f"[Distill] Failed to load checkpoint '{ckpt_path}': {e}")

    def on_fit_start(self) -> None:
        """Retry distillation checkpoint load once Trainer/Datamodule are initialized (for cross-val)."""
        self._init_distillation_from_ckpt(self.model_config)
    
    def _calculate_moe_load_balance_loss(self) -> torch.Tensor:
        """Calculate total load balancing loss from all MoE layers."""
        if not self.has_moe:
            return torch.tensor(0.0, device=self.device)
        
        total_load_balance_loss = torch.tensor(0.0, device=self.device)
        moe_layers = [module for module in self.model.modules() if isinstance(module, MoELayer)]
        
        for moe_layer in moe_layers:
            total_load_balance_loss += moe_layer.get_load_balance_loss()

        total_load_balance_loss/=len(moe_layers)
        
        return total_load_balance_loss
    
    def _calculate_moe_distillation_loss(self) -> torch.Tensor:
        """Aggregate distillation losses from every module that exposes them."""
        if not self.has_moe:
            return None, None, torch.tensor(0.0, device=self.device)
        
        moe_distill_loss = torch.zeros((), device=self.device)
        moe_layers = [module for module in self.model.modules() if isinstance(module, MoELayer)]
        if moe_layers:
            for moe_layer in moe_layers:
                moe_distill_loss += moe_layer.get_distillation_loss()
            
            moe_distill_loss/=len(moe_layers)

        hi_moe_distill_loss = torch.zeros((), device=self.device)
        hi_moe_layers = [module for module in self.model.modules() if isinstance(module, MoelayerRouteFromBoth)]
        if hi_moe_layers:
            for hi_moe_layer in hi_moe_layers:
                hi_moe_distill_loss += hi_moe_layer.get_distillation_loss()

            hi_moe_distill_loss/=len(hi_moe_layers)
        
        total_distill_loss=moe_distill_loss+hi_moe_distill_loss

        return moe_distill_loss, hi_moe_distill_loss, total_distill_loss
    
    def _get_moe_metrics(self) -> Dict[str, torch.Tensor]:
        """Get MoE-specific metrics."""
        if not self.has_moe:
            return {}
        
        metrics = {}
        moe_layers = [module for module in self.model.modules() if isinstance(module, MoELayer)]
        distill_modules = [module for module in self.model.modules() if hasattr(module, "get_distillation_statistics")]
        
        # Calculate average expert utilization
        total_utilization = 0.0
        for i, moe_layer in enumerate(moe_layers):
            load_balance_loss = moe_layer.get_load_balance_loss()
            metrics[f'moe/layer_{i}_load_balance_loss'] = load_balance_loss
            total_utilization += load_balance_loss
        
        metrics['moe/avg_load_balance_loss'] = total_utilization / len(moe_layers) if moe_layers else torch.tensor(0.0)
        metrics['moe/num_moe_layers'] = torch.tensor(len(moe_layers))
        moe_distill, hi_distill, total_distill = self._calculate_moe_distillation_loss()
        metrics['moe/distillation_moe'] = moe_distill
        metrics['moe/distillation_hi'] = hi_distill
        metrics['moe/distillation_total'] = total_distill
        
        
        for idx, module in enumerate(distill_modules):
            stats = module.get_distillation_statistics()
            ce = stats.get("distill_ce")
            kl = stats.get("distill_kl")
            if ce is not None:
                metrics[f'moe/distill_module_{idx}_ce'] = ce
            if kl is not None:
                metrics[f'moe/distill_module_{idx}_kl'] = kl
        
        return metrics
    
    def _log_moe_distributions(self, stage: str, batch_idx: int, on_step: bool, on_epoch: bool) -> None:
        """
        Push detailed router statistics for each MoE layer to TensorBoard.
        """
        if not (self.has_moe and self.logger):
            return
        
        moe_layers = [module for module in self.model.modules() if isinstance(module, MoELayer)]
        experiment = getattr(self.logger, "experiment", None)
        gate_modules = [module for module in self.model.modules() if hasattr(module, "get_gate_statistics")]
        should_log_hist = False
        if stage == "train" and on_step and self.moe_hist_every_n_steps > 0:
            should_log_hist = (self.global_step % self.moe_hist_every_n_steps) == 0
        elif stage != "train" and batch_idx == 0:
            should_log_hist = True
        
        for layer_idx, moe_layer in enumerate(moe_layers):
            stats = moe_layer.get_router_statistics()
            router_probs = stats.get("router_probs")
            topk_probs = stats.get("topk_probs")
            top1_probs = stats.get("top1_probs")
            avg_router_probs = stats.get("avg_router_probs")
            expert_fractions = stats.get("expert_fractions")
            router_entropy = stats.get("router_entropy")
            
            if avg_router_probs is not None:
                for expert_idx, value in enumerate(avg_router_probs):
                    self.log(
                        f"{stage}/moe/layer_{layer_idx}_avg_router_prob_expert_{expert_idx}",
                        value.detach(),
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=False,
                    )
            if expert_fractions is not None:
                for expert_idx, value in enumerate(expert_fractions):
                    self.log(
                        f"{stage}/moe/layer_{layer_idx}_expert_fraction_{expert_idx}",
                        value.detach(),
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=False,
                    )
            if router_entropy is not None:
                self.log(
                    f"{stage}/moe/layer_{layer_idx}_router_entropy",
                    router_entropy.detach(),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=False,
                )
            if top1_probs is not None:
                self.log(
                    f"{stage}/moe/layer_{layer_idx}_top1_prob_mean",
                    top1_probs.mean().detach(),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=False,
                )
            
            if should_log_hist and router_probs is not None and experiment is not None and hasattr(experiment, "add_histogram"):
                hist_step = self.global_step if stage == "train" else self.current_epoch
                experiment.add_histogram(
                    f"{stage}/moe/layer_{layer_idx}_router_probs",
                    router_probs.detach().flatten().cpu(),
                    global_step=hist_step,
                )
            if should_log_hist and topk_probs is not None and experiment is not None and hasattr(experiment, "add_histogram"):
                hist_step = self.global_step if stage == "train" else self.current_epoch
                experiment.add_histogram(
                    f"{stage}/moe/layer_{layer_idx}_topk_probs",
                    topk_probs.detach().flatten().cpu(),
                    global_step=hist_step,
                )
            if should_log_hist and top1_probs is not None and experiment is not None and hasattr(experiment, "add_histogram"):
                hist_step = self.global_step if stage == "train" else self.current_epoch
                experiment.add_histogram(
                    f"{stage}/moe/layer_{layer_idx}_top1_probs",
                    top1_probs.detach().flatten().cpu(),
                    global_step=hist_step,
                )
        
        for gate_idx, gate_module in enumerate(gate_modules):
            stats = gate_module.get_gate_statistics()
            prob_gate = stats.get("prob_gate")
            if prob_gate is None:
                continue
            self.log(
                f"{stage}/moe/gate_{gate_idx}_prob_gate_mean",
                prob_gate.mean().detach(),
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
            )
            if should_log_hist and experiment is not None and hasattr(experiment, "add_histogram"):
                hist_step = self.global_step if stage == "train" else self.current_epoch
                experiment.add_histogram(
                    f"{stage}/moe/gate_{gate_idx}_prob_gate",
                    prob_gate.detach().flatten().cpu(),
                    global_step=hist_step,
                )
    
    def _should_update_prog_bar(self) -> bool:
        """Check if enough time has passed to update progress bar."""
        return False
        # current_time = time.time()
        # if current_time - self.last_prog_bar_update >= self.prog_bar_throttle_interval:
        #     self.last_prog_bar_update = current_time
        #     return True
        # return False
        # return True
    
    def _unpack_batch(self, batch):
        """Normalize batch structure across datasets."""
        if isinstance(batch, dict):
            x = batch.get("image") or batch.get("images") or batch.get("x")
            y = batch.get("label") or batch.get("labels") or batch.get("target") or batch.get("y")
            gaze_map = batch.get("gaze_map") or batch.get("gaze") or batch.get("gazes")
            if x is None or y is None:
                raise ValueError("Batch dictionary must contain image and label entries.")
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                x, gaze_map, y = batch
            elif len(batch) == 2:
                x, y = batch
                gaze_map = None
            else:
                raise ValueError(f"Unexpected batch format with length {len(batch)}.")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        return x, gaze_map, y
    
    def forward(self, x: torch.Tensor, gaze_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        if gaze_map is not None:
            return self.model(x, gaze_map)
        return self.model(x)
    
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, gaze_map, y = self._unpack_batch(batch)
        logits = self.forward(x, gaze_map)
        
        # Calculate main classification loss
        classification_loss = self.criterion(logits, y)
        
        # Calculate MoE load balancing loss if applicable
        moe_load_balance_loss = self._calculate_moe_load_balance_loss()
        _, _, moe_distillation_loss = self._calculate_moe_distillation_loss()
        
        # Combine losses
        if self.has_moe:
            load_balance_weight = self.moe_config['load_balance_weight']
            distillation_weight = self.moe_config['distillation_weight']
            total_loss = (
                classification_loss
                + load_balance_weight * moe_load_balance_loss
                + distillation_weight * moe_distillation_loss
            )
        else:
            total_loss = classification_loss
        
        # Calculate probabilities for metrics
        probs = F.softmax(logits, dim=1)
        
        # Update metrics
        self.train_accuracy(logits, y)
        self.train_precision(logits, y)
        self.train_recall(logits, y)
        self.train_f1(logits, y)
        self.train_auc(probs, y)
        for class_idx, class_accuracy in enumerate(self.train_accuracy_per_class(logits, y)):
            self.log(
                f"train/accuracy_class_{class_idx}",
                class_accuracy.detach(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        for class_idx, class_precision in enumerate(self.train_precision_per_class(logits, y)):
            self.log(
                f"train/precision_class_{class_idx}",
                class_precision.detach(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        for class_idx, class_recall in enumerate(self.train_recall_per_class(logits, y)):
            self.log(
                f"train/recall_class_{class_idx}",
                class_recall.detach(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        for class_idx, class_f1 in enumerate(self.train_f1_per_class(logits, y)):
            self.log(
                f"train/f1_class_{class_idx}",
                class_f1.detach(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        if self.train_auc_per_class is not None:
            per_class_auc = self.train_auc_per_class(probs, y)
            for class_idx, class_auc in enumerate(per_class_auc):
                self.log(
                    f"train/auc_class_{class_idx}",
                    class_auc.detach(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                )
        
        # Check if we should update progress bar
        update_prog_bar = self._should_update_prog_bar()
        
        # Log metrics
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=update_prog_bar)
        self.log("train/classification_loss", classification_loss, on_step=True, on_epoch=True, prog_bar=update_prog_bar)
        self.log("train/accuracy", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=update_prog_bar)
        self.log("train/precision", self.train_precision, on_step=True, on_epoch=True, prog_bar=update_prog_bar)
        self.log("train/recall", self.train_recall, on_step=True, on_epoch=True, prog_bar=update_prog_bar)
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=update_prog_bar)
        self.log("train/auc", self.train_auc, on_step=True, on_epoch=True, prog_bar=update_prog_bar)
        
        # Log MoE-specific metrics
        if self.has_moe:
            self.log("train/moe_load_balance_loss", moe_load_balance_loss, on_step=True, on_epoch=True, prog_bar=update_prog_bar)
            self.log("train/moe_distillation_loss", moe_distillation_loss, on_step=True, on_epoch=True, prog_bar=update_prog_bar)
            moe_metrics = self._get_moe_metrics()
            for metric_name, metric_value in moe_metrics.items():
                self.log(f"train/{metric_name}", metric_value, on_step=True, on_epoch=True, prog_bar=update_prog_bar)
            self._log_moe_distributions("train", batch_idx, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, gaze_map, y = self._unpack_batch(batch)
        logits = self.forward(x, gaze_map)
        
        # Calculate main classification loss
        classification_loss = self.criterion(logits, y)
        
        # Calculate MoE load balancing loss if applicable
        moe_load_balance_loss = self._calculate_moe_load_balance_loss()
        _, _, moe_distillation_loss = self._calculate_moe_distillation_loss()
        
        # Combine losses
        if self.has_moe:
            load_balance_weight = self.moe_config['load_balance_weight']
            distillation_weight = self.moe_config['distillation_weight']
            total_loss = (
                classification_loss
                + load_balance_weight * moe_load_balance_loss
                + distillation_weight * moe_distillation_loss
            )
        else:
            total_loss = classification_loss
        
        # Calculate probabilities for metrics
        probs = F.softmax(logits, dim=1)
        
        # Update metrics
        self.val_accuracy(logits, y)
        self.val_precision(logits, y)
        self.val_recall(logits, y)
        self.val_f1(logits, y)
        self.val_auc(probs, y)
        for class_idx, class_accuracy in enumerate(self.val_accuracy_per_class(logits, y)):
            self.log(
                f"val/accuracy_class_{class_idx}",
                class_accuracy.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        for class_idx, class_precision in enumerate(self.val_precision_per_class(logits, y)):
            self.log(
                f"val/precision_class_{class_idx}",
                class_precision.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        for class_idx, class_recall in enumerate(self.val_recall_per_class(logits, y)):
            self.log(
                f"val/recall_class_{class_idx}",
                class_recall.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        for class_idx, class_f1 in enumerate(self.val_f1_per_class(logits, y)):
            self.log(
                f"val/f1_class_{class_idx}",
                class_f1.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        if self.val_auc_per_class is not None:
            per_class_auc = self.val_auc_per_class(probs, y)
            for class_idx, class_auc in enumerate(per_class_auc):
                self.log(
                    f"val/auc_class_{class_idx}",
                    class_auc.detach(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        if self.moe_config['distillation_active']:
            with torch.no_grad():
                logits_no_gaze = self.forward(x, None)
            no_gaze_loss = self.criterion(logits_no_gaze, y)
            probs_no_gaze = F.softmax(logits_no_gaze, dim=1)
            self.val_accuracy_no_gaze(logits_no_gaze, y)
            self.val_precision_no_gaze(logits_no_gaze, y)
            self.val_recall_no_gaze(logits_no_gaze, y)
            self.val_f1_no_gaze(logits_no_gaze, y)
            self.val_auc_no_gaze(probs_no_gaze, y)
            self.log(
                "val/no_gaze_loss",
                no_gaze_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "val/no_gaze_accuracy",
                self.val_accuracy_no_gaze,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "val/no_gaze_precision",
                self.val_precision_no_gaze,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "val/no_gaze_recall",
                self.val_recall_no_gaze,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "val/no_gaze_f1",
                self.val_f1_no_gaze,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "val/no_gaze_auc",
                self.val_auc_no_gaze,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        
        # Check if we should update progress bar
        update_prog_bar = self._should_update_prog_bar()
        
        # Log metrics
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
        self.log("val/classification_loss", classification_loss, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
        
        # Log MoE-specific metrics
        if self.has_moe:
            self.log("val/moe_load_balance_loss", moe_load_balance_loss, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
            self.log("val/moe_distillation_loss", moe_distillation_loss, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
            moe_metrics = self._get_moe_metrics()
            for metric_name, metric_value in moe_metrics.items():
                self.log(f"val/{metric_name}", metric_value, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
            self._log_moe_distributions("val", batch_idx, on_step=False, on_epoch=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Test step."""
        x, gaze_map, y = self._unpack_batch(batch)
        logits = self.forward(x, gaze_map)
        
        # Calculate main classification loss
        classification_loss = self.criterion(logits, y)
        
        # Calculate MoE load balancing loss if applicable
        moe_load_balance_loss = self._calculate_moe_load_balance_loss()
        _, _, moe_distillation_loss = self._calculate_moe_distillation_loss()
        
        # Combine losses
        if self.has_moe:
            load_balance_weight = self.moe_config['load_balance_weight']
            distillation_weight = self.moe_config['distillation_weight']
            total_loss = (
                classification_loss
                + load_balance_weight * moe_load_balance_loss
                + distillation_weight * moe_distillation_loss
            )
        else:
            total_loss = classification_loss
        
        # Calculate probabilities and predictions
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Store for final evaluation
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        self.test_probabilities.extend(probs.cpu().numpy())
        
        # Update metrics
        self.test_accuracy(logits, y)
        self.test_precision(logits, y)
        self.test_recall(logits, y)
        self.test_f1(logits, y)
        self.test_auc(probs, y)
        for class_idx, class_accuracy in enumerate(self.test_accuracy_per_class(logits, y)):
            self.log(
                f"test/accuracy_class_{class_idx}",
                class_accuracy.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        for class_idx, class_precision in enumerate(self.test_precision_per_class(logits, y)):
            self.log(
                f"test/precision_class_{class_idx}",
                class_precision.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        for class_idx, class_recall in enumerate(self.test_recall_per_class(logits, y)):
            self.log(
                f"test/recall_class_{class_idx}",
                class_recall.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        for class_idx, class_f1 in enumerate(self.test_f1_per_class(logits, y)):
            self.log(
                f"test/f1_class_{class_idx}",
                class_f1.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        if self.test_auc_per_class is not None:
            per_class_auc = self.test_auc_per_class(probs, y)
            for class_idx, class_auc in enumerate(per_class_auc):
                self.log(
                    f"test/auc_class_{class_idx}",
                    class_auc.detach(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        if False and self.moe_config['distillation_active']:
            with torch.no_grad():
                logits_no_gaze = self.forward(x, None)
            no_gaze_loss = self.criterion(logits_no_gaze, y)
            probs_no_gaze = F.softmax(logits_no_gaze, dim=1)
            self.test_accuracy_no_gaze(logits_no_gaze, y)
            self.test_precision_no_gaze(logits_no_gaze, y)
            self.test_recall_no_gaze(logits_no_gaze, y)
            self.test_f1_no_gaze(logits_no_gaze, y)
            self.test_auc_no_gaze(probs_no_gaze, y)
            self.log(
                "test/no_gaze_loss",
                no_gaze_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "test/no_gaze_accuracy",
                self.test_accuracy_no_gaze,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "test/no_gaze_precision",
                self.test_precision_no_gaze,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "test/no_gaze_recall",
                self.test_recall_no_gaze,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "test/no_gaze_f1",
                self.test_f1_no_gaze,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "test/no_gaze_auc",
                self.test_auc_no_gaze,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        
        # Check if we should update progress bar
        update_prog_bar = self._should_update_prog_bar()
        
        # Log metrics
        self.log("test/loss", total_loss, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
        self.log("test/classification_loss", classification_loss, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
        
        # Log MoE-specific metrics
        if self.has_moe:
            self.log("test/moe_load_balance_loss", moe_load_balance_loss, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
            self.log("test/moe_distillation_loss", moe_distillation_loss, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
            moe_metrics = self._get_moe_metrics()
            for metric_name, metric_value in moe_metrics.items():
                self.log(f"test/{metric_name}", metric_value, on_step=False, on_epoch=True, prog_bar=update_prog_bar)
            self._log_moe_distributions("test", batch_idx, on_step=False, on_epoch=True)
        
        return total_loss
    
    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        # Calculate additional metrics
        predictions = np.array(self.test_predictions)
        targets = np.array(self.test_targets)
        probabilities = np.array(self.test_probabilities)
        
        # Classification report
        class_names = [f"Class_{i}" for i in range(self.num_classes)]
        report = classification_report(
            targets, predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Save detailed results
        results = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "predictions": predictions.tolist(),
            "targets": targets.tolist(),
            "probabilities": probabilities.tolist()
        }
        
        # Save to file
        if self.logger:
            log_dir = Path(self.logger.log_dir) if hasattr(self.logger, 'log_dir') else Path("logs")
        else:
            log_dir = Path("logs")
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        with open(log_dir / "test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Log confusion matrix
        self.logger.experiment.add_figure(
            "confusion_matrix", 
            self._plot_confusion_matrix(cm, class_names),
            global_step=self.current_epoch
        )
        
        if self.test_auc_per_class is not None:
            self.test_auc_per_class.reset()
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: list):
        """Plot confusion matrix."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        return plt.gcf()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.training_config.get('epochs', 100)
            )
        elif self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.training_config.get('epochs', 100) // 3, gamma=0.1
            )
        elif self.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=5, factor=0.5
            )
        else:
            return optimizer
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/accuracy" if self.scheduler_type == "plateau" else None
            }
        }
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Reset metrics for next epoch
        # input(0)
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()
        self.train_accuracy_per_class.reset()
        self.train_precision_per_class.reset()
        self.train_recall_per_class.reset()
        self.train_f1_per_class.reset()
        # self.train_auc.reset()
        if self.train_auc_per_class is not None:
            self.train_auc_per_class.reset()
        # input(1)
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Reset metrics for next epoch
        # input(2)
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_accuracy_per_class.reset()
        self.val_precision_per_class.reset()
        self.val_recall_per_class.reset()
        self.val_f1_per_class.reset()
        # self.val_auc.reset()
        if self.val_auc_per_class is not None:
            self.val_auc_per_class.reset()
        # input(3)
    
    def on_test_epoch_start(self) -> None:
        """Called at the start of test epoch."""
        # Clear stored predictions
        # input(4)
        self.test_predictions = []
        self.test_targets = []
        self.test_probabilities = []
        self.test_accuracy_per_class.reset()
        self.test_precision_per_class.reset()
        self.test_recall_per_class.reset()
        self.test_f1_per_class.reset()
        # input(5)
