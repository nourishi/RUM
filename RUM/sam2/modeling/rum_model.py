import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyMemoryScheduler(nn.Module):
    """Predict tracking uncertainty and soft control weights."""

    def __init__(
        self,
        hidden_dim: int,
        write_threshold: float = 0.35,
        relocalize_threshold: float = 0.65,
        control_temperature: float = 0.7,
    ):
        super().__init__()
        mlp_dim = max(hidden_dim // 4, 16)
        self.score_head = nn.Sequential(
            nn.Linear(3, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, 1),
        )
        self.action_head = nn.Sequential(
            nn.Linear(4, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 3),
        )
        self.write_threshold = write_threshold
        self.relocalize_threshold = relocalize_threshold
        self.control_temperature = max(float(control_temperature), 1e-3)

    def forward(self, obj_conf, iou_conf, proto_conf):
        features = torch.cat([obj_conf, iou_conf, proto_conf], dim=-1)
        residual = 0.1 * torch.tanh(self.score_head(features))
        uncertainty = (
            1.0 - (0.45 * obj_conf + 0.45 * iou_conf + 0.10 * proto_conf) + residual
        ).clamp(0.0, 1.0)

        policy_input = torch.cat([features, uncertainty], dim=-1)
        action_logits = self.action_head(policy_input)
        midpoint = 0.5 * (self.write_threshold + self.relocalize_threshold)
        action_bias = torch.cat(
            [
                self.write_threshold - uncertainty,
                -torch.abs(uncertainty - midpoint),
                uncertainty - self.relocalize_threshold,
            ],
            dim=-1,
        )
        action_logits = action_logits + 3.0 * action_bias
        action_probs = torch.softmax(
            action_logits / self.control_temperature, dim=-1
        )
        return (
            uncertainty,
            action_probs[:, 0:1],
            action_probs[:, 1:2],
            action_probs[:, 2:3],
        )


class VisualPrototypeMemory(nn.Module):
    """Maintain short-term and long-term visual prototypes."""

    def __init__(
        self,
        hidden_dim: int,
        num_parts: int = 4,
        momentum: float = 0.95,
        short_momentum: float = 0.6,
        long_momentum: float = 0.98,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_parts = num_parts
        self.momentum = momentum
        self.short_momentum = short_momentum
        self.long_momentum = long_momentum

        self.global_fuser = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.part_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ptr_adapter = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        blend_dim = max(hidden_dim // 8, 8)
        self.blend_head = nn.Sequential(
            nn.Linear(2, blend_dim),
            nn.ReLU(inplace=True),
            nn.Linear(blend_dim, 1),
        )
        self.memory_selector = nn.Sequential(
            nn.Linear(3, blend_dim),
            nn.ReLU(inplace=True),
            nn.Linear(blend_dim, 1),
        )

    def _masked_pool(self, feat, weight):
        denom = weight.sum(dim=(2, 3)).clamp_min(1e-6)
        pooled = (feat * weight).sum(dim=(2, 3)) / denom
        return pooled

    def _quadrant_masks(self, h, w, device, dtype):
        ys = torch.arange(h, device=device)
        xs = torch.arange(w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        y_half = h // 2
        x_half = w // 2
        quadrants = [
            (grid_y < y_half) & (grid_x < x_half),
            (grid_y < y_half) & (grid_x >= x_half),
            (grid_y >= y_half) & (grid_x < x_half),
            (grid_y >= y_half) & (grid_x >= x_half),
        ]
        return torch.stack([q.to(dtype=dtype) for q in quadrants], dim=0).unsqueeze(1)

    def _proto_similarity(self, obj_ptr, current_part_proto, bank_global, bank_part, valid):
        global_sim = F.cosine_similarity(
            obj_ptr.float(), bank_global.float(), dim=-1
        ).unsqueeze(-1)
        part_sim = F.cosine_similarity(
            current_part_proto.float(), bank_part.float(), dim=-1
        ).mean(dim=1, keepdim=True)
        sim = 0.7 * global_sim + 0.3 * part_sim
        sim = 0.5 * (sim + 1.0)
        return sim.clamp(0.0, 1.0) * valid.float()

    def _refresh_combined_state(self, state):
        short_valid = state["short_valid"].float()
        long_valid = state["long_valid"].float()
        denom = (short_valid + long_valid).clamp_min(1.0)
        state["global_proto"] = (
            short_valid * state["short_global_proto"]
            + long_valid * state["long_global_proto"]
        ) / denom
        state["part_proto"] = (
            short_valid.unsqueeze(-1) * state["short_part_proto"]
            + long_valid.unsqueeze(-1) * state["long_part_proto"]
        ) / denom.unsqueeze(-1)
        state["proto_valid"] = (short_valid + long_valid) > 0

    def _apply_update(self, state, prefix, global_proto, part_proto, update_weight, momentum):
        global_key = f"{prefix}_global_proto"
        part_key = f"{prefix}_part_proto"
        valid_key = f"{prefix}_valid"
        weight = update_weight.float().clamp(0.0, 1.0)
        valid = state[valid_key].float()

        old_global = state[global_key]
        old_part = state[part_key]
        blended_global = valid * (
            momentum * old_global + (1.0 - momentum) * global_proto
        ) + (1.0 - valid) * global_proto
        blended_part = valid.unsqueeze(-1) * (
            momentum * old_part + (1.0 - momentum) * part_proto
        ) + (1.0 - valid).unsqueeze(-1) * part_proto

        state[global_key] = (1.0 - weight) * old_global + weight * blended_global
        state[part_key] = (1.0 - weight).unsqueeze(-1) * old_part + weight.unsqueeze(
            -1
        ) * blended_part
        state[valid_key] = torch.logical_or(state[valid_key], weight > 1e-4)

    def extract(self, obj_ptr, pix_feat, pred_mask_logits):
        b, _, h, w = pix_feat.shape
        mask_prob = torch.sigmoid(pred_mask_logits)
        if mask_prob.shape[-2:] != (h, w):
            mask_prob = F.interpolate(
                mask_prob, size=(h, w), mode="bilinear", align_corners=False
            )

        pooled_feat = self._masked_pool(pix_feat, mask_prob)
        global_proto = self.global_fuser(torch.cat([obj_ptr, pooled_feat], dim=-1))

        quad_masks = self._quadrant_masks(h, w, pix_feat.device, pix_feat.dtype)
        part_tokens = []
        for idx in range(self.num_parts):
            quad_idx = idx % quad_masks.shape[0]
            weight = mask_prob * quad_masks[quad_idx : quad_idx + 1]
            part_tokens.append(self._masked_pool(pix_feat, weight))
        part_proto = torch.stack(part_tokens, dim=1)
        part_proto = self.part_proj(part_proto)
        return global_proto, part_proto

    def similarity(self, state, obj_ptr, current_part_proto):
        if state is None:
            return torch.zeros(obj_ptr.size(0), 1, device=obj_ptr.device)

        self._refresh_combined_state(state)
        short_sim = self._proto_similarity(
            obj_ptr,
            current_part_proto,
            state["short_global_proto"],
            state["short_part_proto"],
            state["short_valid"],
        )
        long_sim = self._proto_similarity(
            obj_ptr,
            current_part_proto,
            state["long_global_proto"],
            state["long_part_proto"],
            state["long_valid"],
        )
        selector_input = torch.cat(
            [short_sim, long_sim, (short_sim - long_sim).abs()], dim=-1
        )
        long_weight = torch.sigmoid(self.memory_selector(selector_input))
        valid = torch.logical_or(state["short_valid"], state["long_valid"]).float()
        sim = (1.0 - long_weight) * short_sim + long_weight * long_sim
        return (sim * valid).clamp(0.0, 1.0)

    def get_memory_pointer(self, state):
        if state is None:
            return None
        self._refresh_combined_state(state)
        if not state["proto_valid"].any():
            return None
        proto_summary = torch.cat(
            [state["global_proto"], state["part_proto"].mean(dim=1)], dim=-1
        )
        return self.ptr_adapter(proto_summary)

    def refine_obj_ptr(self, obj_ptr, state, uncertainty):
        if state is None:
            return obj_ptr

        self._refresh_combined_state(state)
        short_valid = state["short_valid"].float()
        long_valid = state["long_valid"].float()
        short_summary = torch.cat(
            [state["short_global_proto"], state["short_part_proto"].mean(dim=1)], dim=-1
        )
        long_summary = torch.cat(
            [state["long_global_proto"], state["long_part_proto"].mean(dim=1)], dim=-1
        )
        short_ptr = self.ptr_adapter(short_summary)
        long_ptr = self.ptr_adapter(long_summary)

        short_sim = 0.5 * (
            F.cosine_similarity(obj_ptr.float(), short_ptr.float(), dim=-1).unsqueeze(-1)
            + 1.0
        ) * short_valid
        long_sim = 0.5 * (
            F.cosine_similarity(obj_ptr.float(), long_ptr.float(), dim=-1).unsqueeze(-1)
            + 1.0
        ) * long_valid
        selector_input = torch.cat([uncertainty, short_sim, long_sim], dim=-1)
        long_weight = torch.sigmoid(self.memory_selector(selector_input))
        proto_ptr = (1.0 - long_weight) * short_ptr + long_weight * long_ptr
        sim = torch.maximum(short_sim, long_sim)
        blend = torch.sigmoid(self.blend_head(torch.cat([uncertainty, sim], dim=-1)))
        valid = torch.logical_or(state["short_valid"], state["long_valid"]).float()
        blend = blend * valid
        return (1.0 - blend) * obj_ptr + blend * proto_ptr

    def update(self, state, global_proto, part_proto, update_weight, uncertainty):
        if state is None:
            return

        self._apply_update(
            state,
            "short",
            global_proto,
            part_proto,
            update_weight,
            self.short_momentum,
        )
        long_weight = update_weight.float().clamp(0.0, 1.0) * (1.0 - uncertainty).clamp(
            0.0, 1.0
        )
        self._apply_update(
            state,
            "long",
            global_proto,
            part_proto,
            long_weight,
            self.long_momentum,
        )
        self._refresh_combined_state(state)


class MotionRelocalizer(nn.Module):
    """Generate a region-level spatial prior for re-localization."""

    def __init__(
        self,
        hidden_dim: int,
        prior_strength: float = 3.0,
        prior_sigma: float = 0.12,
        max_delta: float = 0.2,
        min_region_scale: float = 0.06,
        max_region_scale: float = 0.35,
    ):
        super().__init__()
        self.prior_strength = prior_strength
        self.prior_sigma = prior_sigma
        self.max_delta = max_delta
        self.min_region_scale = min_region_scale
        self.max_region_scale = max_region_scale

        mlp_dim = max(hidden_dim // 8, 8)
        self.prior_gate = nn.Sequential(
            nn.Linear(2, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, 1),
        )

    def compute_mask_region(self, mask_logits):
        prob = torch.sigmoid(mask_logits).squeeze(1)
        b, h, w = prob.shape
        mass = prob.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)

        ys = torch.linspace(0.0, 1.0, h, device=prob.device, dtype=prob.dtype)
        xs = torch.linspace(0.0, 1.0, w, device=prob.device, dtype=prob.dtype)
        grid_y = ys.view(1, h, 1)
        grid_x = xs.view(1, 1, w)

        cy = (prob * grid_y).sum(dim=(1, 2), keepdim=True) / mass
        cx = (prob * grid_x).sum(dim=(1, 2), keepdim=True) / mass
        center = torch.cat([cx, cy], dim=-1).view(b, 2)

        var_y = (prob * (grid_y - cy.view(b, 1, 1)) ** 2).sum(dim=(1, 2), keepdim=True) / mass
        var_x = (prob * (grid_x - cx.view(b, 1, 1)) ** 2).sum(dim=(1, 2), keepdim=True) / mass
        size_y = (2.5 * torch.sqrt(var_y.clamp_min(1e-6))).clamp(
            self.min_region_scale, self.max_region_scale
        )
        size_x = (2.5 * torch.sqrt(var_x.clamp_min(1e-6))).clamp(
            self.min_region_scale, self.max_region_scale
        )
        size = torch.cat([size_x, size_y], dim=-1).view(b, 2)
        return center, size

    def compute_mask_center(self, mask_logits):
        center, _ = self.compute_mask_region(mask_logits)
        return center

    def predict_center(self, last_center, prev_center):
        delta = (last_center - prev_center).clamp(-self.max_delta, self.max_delta)
        return (last_center + delta).clamp(0.0, 1.0)

    def predict_region(self, last_center, prev_center, last_size, prev_size):
        pred_center = self.predict_center(last_center, prev_center)
        size_delta = (last_size - prev_size).clamp(
            -0.5 * self.max_delta, 0.5 * self.max_delta
        )
        pred_size = (last_size + size_delta).clamp(
            self.min_region_scale, self.max_region_scale
        )
        return pred_center, pred_size

    def _region_prior(self, centers, sizes, h, w, dtype):
        ys = torch.linspace(0.0, 1.0, h, device=centers.device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, w, device=centers.device, dtype=dtype)
        grid_y = ys.view(1, h, 1)
        grid_x = xs.view(1, 1, w)

        center_x = centers[:, 0].view(-1, 1, 1)
        center_y = centers[:, 1].view(-1, 1, 1)
        sigma_x = (0.5 * sizes[:, 0].view(-1, 1, 1) + self.prior_sigma).clamp_min(1e-4)
        sigma_y = (0.5 * sizes[:, 1].view(-1, 1, 1) + self.prior_sigma).clamp_min(1e-4)
        dist2 = ((grid_x - center_x) / sigma_x) ** 2 + ((grid_y - center_y) / sigma_y) ** 2
        gaussian = torch.exp(-0.5 * dist2)
        half_w = 0.5 * sizes[:, 0].view(-1, 1, 1)
        half_h = 0.5 * sizes[:, 1].view(-1, 1, 1)
        box = (
            (grid_x - center_x).abs() <= half_w
        ) & ((grid_y - center_y).abs() <= half_h)
        return (0.7 * gaussian + 0.3 * box.to(dtype=dtype)).clamp(0.0, 1.0)

    def apply_motion_prior(
        self,
        mask_logits,
        predicted_center,
        predicted_size,
        uncertainty,
        relocalize_weight=None,
    ):
        b, _, h, w = mask_logits.shape
        prior = self._region_prior(
            predicted_center, predicted_size, h, w, mask_logits.dtype
        )
        if relocalize_weight is None:
            relocalize_weight = torch.ones_like(uncertainty)
        gate_input = torch.cat([uncertainty, relocalize_weight], dim=-1)
        gate = torch.sigmoid(self.prior_gate(gate_input)).view(b, 1, 1, 1)
        boost = self.prior_strength * gate * (prior.unsqueeze(1) - 0.5)
        return mask_logits + boost
