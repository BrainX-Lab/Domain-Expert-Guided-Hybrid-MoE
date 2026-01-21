"""
General Mixture of Experts (MoE) layer implementation.

This module provides a flexible MoE layer that can work with any expert and gating classes.
"""

from typing import Any, Dict, Type, Union, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .helper import IgnoreOtherInputWarpper as IOIW, MoEWarpper as MOEW

def MLP(in_features: int, out_features: int, hidden_dim: Optional[int] = None):
    """Compact MLP used for student gating networks."""
    if hidden_dim is None:
        hidden_dim = torch.max(128, in_features)
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_features),
    )


class MoELayer(nn.Module):

    def __init__(
        self,
        expert_class: Type[nn.Module],
        expert_params: Dict[str, Any],
        gating_class: Type[nn.Module],
        gating_params: Dict[str, Any],
        num_experts: int = 4,
        top_k: int = 2,
        # load_balance_weight: float = 0.01,
        gate_input_fn: Callable[[torch.Tensor], torch.Tensor] = None,
        student_gate_class: Optional[Type[nn.Module]] = None,
        student_gate_params: Optional[Dict[str, Any]] = None,
        distillation_active: bool = False,
        **kwargs
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        # self.load_balance_weight = load_balance_weight
        
        # Merge expert_params with kwargs
        expert_kwargs = {**expert_params, **kwargs}
        
        # Create expert networks
        self.experts = nn.ModuleList([
            expert_class(**expert_kwargs)
            for _ in range(num_experts)
        ])
        
        # Create gating network
        self.gate = gating_class(**gating_params)
        
        if hasattr(self.gate, 'weight') and hasattr(self.gate, 'bias'):
            nn.init.normal_(self.gate.weight, std=0.02)
            nn.init.zeros_(self.gate.bias)
        
        # Gate input transformation function
        if gate_input_fn is None:
            # Default: global average pooling for spatial dimensions
            self.gate_input_fn = self._default_gate_input_fn
        else:
            self.gate_input_fn = gate_input_fn

        if student_gate_class is not None and student_gate_params is not None:
            self.student_gate = student_gate_class(**student_gate_params)
        else:
            self.student_gate = None
        self.distillation_active = bool(distillation_active)
        
        # Register load balancing loss as buffer
        self.register_buffer('load_balance_loss', torch.tensor(0.0))
        
        # Internal caches for router statistics captured during forward passes
        self._last_router_probs: Optional[torch.Tensor] = None
        self._last_topk_probs: Optional[torch.Tensor] = None
        self._last_top1_probs: Optional[torch.Tensor] = None
        self._last_topk_indices: Optional[torch.Tensor] = None
        self._last_expert_fractions: Optional[torch.Tensor] = None
        self._last_avg_router_probs: Optional[torch.Tensor] = None
        self._last_router_entropy: Optional[torch.Tensor] = None
        self._distill_total_loss: Optional[torch.Tensor] = None
        self._last_distill_ce: Optional[torch.Tensor] = None
        self._last_distill_kl: Optional[torch.Tensor] = None
        self._last_distill_l1: Optional[torch.Tensor] = None

    def _default_gate_input_fn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Default gate input function: global average pooling for spatial dimensions.
        
        Args:
            x: Input tensor of shape [batch_size, channels, ...]
            
        Returns:
            Gate input tensor of shape [batch_size, channels]
        """
        if x.dim() > 2:
            # Global average pooling over spatial dimensions
            return torch.mean(x, dim=list(range(2, x.dim())))
        else:
            # Already 2D, return as is
            return x
    
    def forward(self, x: torch.Tensor, gate_input: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the MoE layer.

        Args:
            x: Input tensor
            
        Returns:
            Output tensor from selected experts
        """
        batch_size = x.shape[0]
        
        # Prepare student features from x for distillation or fallback routing
        student_logits: Optional[torch.Tensor] = None
        if self.distillation_active and self.student_gate is not None:
            student_features = self.gate_input_fn(x)
            if student_features.dim() > 2:
                student_features = student_features.view(student_features.size(0), -1)
        teacher_logits = None

        if gate_input is not None:
            # Compute expert logits from the trained gate (teacher)
            teacher_logits = self.gate(gate_input)
            expert_logits = teacher_logits
        else:
            # No teacher gate input; only route via student if distillation is active
            # if not self.distillation_active or self.student_gate is None:
            #     raise RuntimeError("Missing gate_input and distillation is inactive; cannot route without teacher gate input.")
            student_hid_feature = self._forward_student_gate(student_features)
            student_logits = self.gate(student_hid_feature)
            expert_logits = student_logits

        # Apply softmax to get expert probabilities
        expert_probs = torch.softmax(expert_logits, dim=-1)  # [batch_size, num_experts]
        self._reset_distillation_cache(expert_probs.device, expert_probs.dtype)
        # Compute distillation losses only when active and both teacher and student features are present
        if self.distillation_active and ('student_features' in locals()) and (gate_input is not None):
            # Pass student/teacher features to compute losses (not logits)
            self._compute_distillation_losses(student_features, gate_input)
            pass

        # Top-k expert selection
        top_k_probs, top_k_indices = torch.topk(expert_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Renormalize

        # Cache router statistics for external logging
        self._last_router_probs = expert_probs.detach()
        self._last_topk_probs = top_k_probs.detach()
        self._last_topk_indices = top_k_indices.detach()
        if self.top_k > 0:
            self._last_top1_probs = top_k_probs[:, 0].detach()
        else:
            self._last_top1_probs = None
        
        # Initialize output - we need to get the output shape from one expert first
        sample_output = self.experts[0](x[:1])  # Get output shape from first expert
        output_shape = sample_output.shape
        output = torch.zeros(batch_size, *output_shape[1:], device=x.device, dtype=x.dtype)
        
        # Route to selected experts - optimized version
        
        # Process each expert only once with all its assigned samples
        for expert_id in range(self.num_experts):
            # Find all samples that use this expert across all top_k selections
            expert_mask = (top_k_indices == expert_id)  # [batch_size, top_k]
            
            if expert_mask.any():
                # Get batch indices and corresponding weights for this expert
                batch_indices = []
                weights = []
                
                for k in range(self.top_k):
                    mask = expert_mask[:, k]  # [batch_size]
                    if mask.any():
                        batch_indices.extend(torch.where(mask)[0].tolist())
                        weights.extend(top_k_probs[mask, k].tolist())
                
                if batch_indices:
                    # Extract samples for this expert
                    expert_samples = x[batch_indices]  # [num_samples_for_expert, ...]
                    
                    # Process all samples for this expert in one forward pass
                    expert_outputs = self.experts[expert_id](expert_samples)  # [num_samples_for_expert, ...]
                    
                    # Accumulate weighted outputs back to original positions
                    for i, (batch_idx, weight) in enumerate(zip(batch_indices, weights)):
                        output[batch_idx] += weight * expert_outputs[i]
        
        # Compute load balancing loss
        self._compute_load_balance_loss(expert_probs, top_k_indices)
        
        return output

    def _forward_student_gate(self, student_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the configured student (distillation) gate."""
        if self.student_gate is None:
            raise RuntimeError(
                "Student gate not configured; provide student_gate_class and student_gate_params to enable distillation."
            )
        return self.student_gate(student_features)

    def _compute_distillation_losses(
        self,
        student_features: torch.Tensor,
        teacher_features: Optional[torch.Tensor],
    ) -> None:
        """
        Compute and cache distillation losses using student/teacher features.

        The features are first passed through their respective gates to obtain logits,
        then the original CE and KL objectives are computed between those logits.
        Additionally, an L1 loss is computed between student logits and detached teacher logits.
        """
        zero = student_features.new_zeros(())
        self._distill_total_loss = zero
        self._last_distill_ce = None
        self._last_distill_kl = None
        self._last_distill_l1 = None

        if teacher_features is None:
            return

        # Derive logits from features via the configured gates
        student_hid_feature = self._forward_student_gate(student_features)
        student_logits = self.gate(student_hid_feature)
        teacher_logits = self.gate(teacher_features)

        teacher_probs = torch.softmax(teacher_logits, dim=-1).detach()
        teacher_top1 = torch.argmax(teacher_probs, dim=-1)
        ce_loss = F.cross_entropy(student_logits, teacher_top1)
        student_log_probs = torch.log_softmax(student_logits, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        l1_loss = F.l1_loss(student_hid_feature, teacher_features.detach())
        total_loss = ce_loss + kl_loss + l1_loss

        self._distill_total_loss = total_loss
        self._last_distill_ce = ce_loss.detach()
        self._last_distill_kl = kl_loss.detach()
        self._last_distill_l1 = l1_loss.detach()

    def _compute_load_balance_loss(self, expert_probs: torch.Tensor, top_k_indices: torch.Tensor):
        """
        Compute load balancing loss to encourage equal expert utilization.
        
        Args:
            expert_probs: Expert probabilities [batch_size, num_experts]
            top_k_indices: Selected expert indices [batch_size, top_k]
        """
        batch_size = expert_probs.size(0)
        if batch_size == 0:
            self.load_balance_loss = torch.zeros((), device=expert_probs.device, dtype=expert_probs.dtype)
            return

        device = expert_probs.device
        dtype = expert_probs.dtype

        # Fraction of tokens routed to each expert (f_i)
        assignment_counts = torch.zeros(self.num_experts, device=device, dtype=dtype)
        flat_indices = top_k_indices.reshape(-1)
        assignment_counts.scatter_add_(0, flat_indices, torch.ones(flat_indices.numel(), device=device, dtype=dtype))
        total_assignments = batch_size * self.top_k
        expert_fractions = assignment_counts / total_assignments if total_assignments > 0 else assignment_counts

        # Average router probability per expert (p_i)
        avg_router_probs = expert_probs.mean(dim=0)

        num_experts = torch.tensor(self.num_experts, device=device, dtype=dtype)
        self.load_balance_loss = num_experts * torch.sum(expert_fractions * avg_router_probs)
        
        # Cache additional routing summaries
        self._last_expert_fractions = expert_fractions.detach()
        self._last_avg_router_probs = avg_router_probs.detach()
        # Small epsilon avoids log(0) without affecting gradients (already detached)
        router_entropy = -(expert_probs * torch.log(expert_probs + 1e-8)).sum(dim=-1).mean()
        self._last_router_entropy = router_entropy.detach()

    def _reset_distillation_cache(self, device: torch.device, dtype: torch.dtype) -> None:
        """Reset distillation-related caches before computing losses.

        Parameters:
            device: Target device for allocated tensors.
            dtype: Target dtype for allocated tensors.
        """
        self._distill_total_loss = torch.zeros((), device=device, dtype=dtype)
        self._last_distill_ce = None
        self._last_distill_kl = None
        self._last_distill_l1 = None

    def get_load_balance_loss(self) -> torch.Tensor:
        """
        Get the current load balancing loss.
        
        Returns:
            Load balancing loss tensor
        """
        return self.load_balance_loss
    
    def get_router_statistics(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieve cached routing statistics computed during the last forward pass.
        """
        return {
            "router_probs": self._last_router_probs,
            "topk_probs": self._last_topk_probs,
            "topk_indices": self._last_topk_indices,
            "expert_fractions": self._last_expert_fractions,
            "avg_router_probs": self._last_avg_router_probs,
            "router_entropy": self._last_router_entropy,
            "top1_probs": self._last_top1_probs,
        }

    def get_distillation_loss(self) -> torch.Tensor:
        """Return the current distillation loss (sum of CE and KL components)."""
        if self._distill_total_loss is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self._distill_total_loss

    def get_distillation_statistics(self) -> Dict[str, Optional[torch.Tensor]]:
        """Retrieve cached distillation statistics from the last forward pass."""
        return {
            "distill_ce": self._last_distill_ce,
            "distill_kl": self._last_distill_kl,
            "distill_l1": self._last_distill_l1,
        }

    def set_distillation_active(self, active: bool) -> None:
        """Enable or disable distillation for this layer at runtime."""
        self.distillation_active = bool(active)

def moelayer_route_from_expert(expert_class, expert_params, img_channels, num_experts, top_k, distillation_active: bool = False):
    student_gate_class = None
    student_gate_params = None
    if distillation_active:
        student_gate_class = MLP
        student_gate_params = {
            'in_features': img_channels,
            'out_features': 128,
            'hidden_dim': max(128, img_channels),
        }
    return MoELayer(
        expert_class=expert_class,
        expert_params=expert_params,
        gating_class=MLP,
        gating_params={
            'in_features': 128,
            'out_features': num_experts,
            'hidden_dim': max(128, img_channels),
        },
        num_experts=num_experts,
        top_k=top_k,
        student_gate_class=student_gate_class,
        student_gate_params=student_gate_params,
        distillation_active=distillation_active,
        # load_balance_weight=load_balance_weight,
        gate_input_fn=None, # default input fn=torch.mean(x, dim=list(range(2, x.dim())))
    )

def moelayer_route_from_image(expert_class, expert_params, img_channels, num_experts, top_k, distillation_active: bool = False):
    student_gate_class = None
    student_gate_params = None
    return MoELayer(
        expert_class=expert_class,
        expert_params=expert_params,
        gating_class = MLP,
        gating_params={
            'in_features': img_channels,
            'out_features': num_experts,
            'hidden_dim': max(128, img_channels),
        },
        num_experts=num_experts,
        top_k=top_k,
        student_gate_class=student_gate_class,
        student_gate_params=student_gate_params,
        distillation_active=False,
        gate_input_fn=None, # default input fn=torch.mean(x, dim=list(range(2, x.dim())))
    )

class MoelayerRouteFromImage(nn.Module):
    def __init__(self, expert_class, expert_params, img_channels, num_experts, top_k, distillation_active):
        super().__init__()
        self.moelayer_route_from_image=moelayer_route_from_image(expert_class, expert_params, img_channels, num_experts, top_k, distillation_active)

    def forward(self, input):
        x, _ = input
        x_feature=torch.mean(x, dim=list(range(2, x.dim()))) if x.dim()>2 else x
        x_out=self.moelayer_route_from_image(x, x_feature)
        return (x_out, _)

class MoelayerRouteFromExpert(nn.Module):
    def __init__(self, expert_class, expert_params, img_channels, num_experts, top_k, distillation_active):
        super().__init__()
        self.moelayer_route_from_expert=moelayer_route_from_expert(expert_class, expert_params, img_channels, num_experts, top_k, distillation_active)

    def forward(self, input):
        x, expert_map_feat = input
        x_out=self.moelayer_route_from_expert(x, expert_map_feat)
        return (x_out, expert_map_feat)
    
class MoelayerRouteFromBoth(nn.Module):
    def __init__(self, expert_class, expert_params, img_channels, num_experts, top_k, distillation_active):
        super().__init__()
        self.moelayer_route_from_image=moelayer_route_from_image(expert_class, expert_params, img_channels, num_experts, top_k)
        self.moelayer_route_from_expert=moelayer_route_from_expert(expert_class, expert_params, img_channels, num_experts, top_k, distillation_active)
        self.porb_gate=nn.Sequential(nn.Linear(img_channels+128, 1), nn.Sigmoid(), nn.Flatten(0))
        self._last_prob_gate: Optional[torch.Tensor] = None
        self._prob_gate_student: Optional[nn.Module] = None
        self._prob_distill_total_loss: Optional[torch.Tensor] = None
        self._last_prob_distill_ce: Optional[torch.Tensor] = None
        self._last_prob_distill_kl: Optional[torch.Tensor] = None
        self.distillation_active = bool(distillation_active)
        if distillation_active:
            self._prob_gate_student=MLP(img_channels, 1, max(128, img_channels))

    def forward(self, input):
        x, expert_map_feat = input
        if x.dim() > 2:
            x_feat=torch.mean(x, dim=list(range(2, x.dim())))
        else:
            print(x.dim())
            x_feat=x

        student_logits = None
        student_prob = None
        if self.distillation_active and (self._prob_gate_student is not None):
            student_logits = self._forward_prob_gate_student(x_feat)
            student_prob = torch.sigmoid(student_logits).view(-1)

        teacher_prob = None
        if expert_map_feat is not None:
            teacher_prob = self.porb_gate(torch.cat([x_feat, expert_map_feat], dim=-1))

        self._reset_prob_distillation_cache(x.device)
        if self.distillation_active and (student_logits is not None) and (student_prob is not None) and (teacher_prob is not None):
            self._compute_prob_gate_distillation_losses(student_logits.view(-1), student_prob, teacher_prob)

        if teacher_prob is not None:
            p = teacher_prob
        elif self.distillation_active and (student_prob is not None):
            p = student_prob
        else:
            raise RuntimeError("Missing teacher prob and distillation is inactive; cannot compute probability gate.")

        self._last_prob_gate = p.detach()
        # Always pass teacher gate input to the image router (it derives from x)
        x_feat_batched = torch.mean(x, dim=list(range(2, x.dim()))) if x.dim() > 2 else x
        x_out_1=self.moelayer_route_from_image(x, x_feat_batched)
        x_out_2=self.moelayer_route_from_expert(x, expert_map_feat)
        p_broadcast=p.view(-1, *([1] * (x.dim() - 1)))
        return (x_out_1*p_broadcast+x_out_2*(1-p_broadcast), expert_map_feat)

    def set_distillation_active(self, active: bool) -> None:
        self.distillation_active = bool(active)
    
    def get_gate_statistics(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieve cached probability gate output from the last forward pass.
        """
        return {"prob_gate": self._last_prob_gate}

    def _forward_prob_gate_student(self, x_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass for the student probability gate that uses image features only."""
        return self._prob_gate_student(x_feat)

    def _compute_prob_gate_distillation_losses(
        self,
        student_logits: torch.Tensor,
        student_prob: torch.Tensor,
        teacher_prob: Optional[torch.Tensor],
    ) -> None:
        """Compute CE and KL distillation objectives for the probability gate."""
        zero = student_logits.new_zeros(())
        self._prob_distill_total_loss = zero
        self._last_prob_distill_ce = None
        self._last_prob_distill_kl = None

        if teacher_prob is None:
            return

        teacher_prob = teacher_prob.detach()
        teacher_prob = teacher_prob.view(-1)
        teacher_top1 = (teacher_prob >= 0.5).float()
        ce_loss = F.binary_cross_entropy_with_logits(student_logits, teacher_top1)

        eps = 1e-8
        student_probs_2 = torch.stack([student_prob, 1 - student_prob], dim=-1).clamp(min=eps, max=1 - eps)
        teacher_probs_2 = torch.stack([teacher_prob, 1 - teacher_prob], dim=-1).clamp(min=eps, max=1 - eps)
        student_log_probs_2 = torch.log(student_probs_2)
        kl_loss = F.kl_div(student_log_probs_2, teacher_probs_2, reduction="batchmean")

        self._prob_distill_total_loss = ce_loss + kl_loss
        self._last_prob_distill_ce = ce_loss.detach()
        self._last_prob_distill_kl = kl_loss.detach()

    def _reset_prob_distillation_cache(self, device: torch.device) -> None:
        """Reset probability gate distillation caches before computing losses."""
        self._prob_distill_total_loss = torch.zeros((), device=device)
        self._last_prob_distill_ce = None
        self._last_prob_distill_kl = None

    def get_distillation_loss(self) -> torch.Tensor:
        """Return the distillation loss contributed by the probability gate."""
        if self._prob_distill_total_loss is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self._prob_distill_total_loss

    def get_distillation_statistics(self) -> Dict[str, Optional[torch.Tensor]]:
        """Expose the CE and KL components from the last probability gate distillation step."""
        return {
            "distill_ce": self._last_prob_distill_ce,
            "distill_kl": self._last_prob_distill_kl,
        }

def get_moetype_class(moetype :str):
    """
    Return the routing factory corresponding to the requested MoE type.
    """
    normalized = (moetype or "").strip().lower()
    routing_factories = {
        "image": MoelayerRouteFromImage,
        "expert": MoelayerRouteFromExpert,
        "both": MoelayerRouteFromBoth,
    }
    try:
        return routing_factories[normalized]
    except KeyError as err:
        valid_keys = ", ".join(routing_factories.keys())
        raise ValueError(f"Unknown MoE router type '{moetype}'. Expected one of: {valid_keys}.") from err
