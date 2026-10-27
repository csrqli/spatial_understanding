"""F0 local and F1 image-scene alignment; no temporal tokens or LLM token expansion."""
import torch
from torch import nn
from torch.nn import functional as F


def masked_grid_pool(patch, valid_weight, size):
    """Pool real patch-grid tokens only, weighting fractional unpadded areas."""
    if patch.ndim != 4 or valid_weight.shape != patch.shape[:3]:
        raise ValueError("Expected patch [B,H,W,D] and valid_weight [B,H,W]")
    weights = valid_weight.to(patch.dtype).unsqueeze(1)
    numerator = F.adaptive_avg_pool2d(patch.permute(0, 3, 1, 2) * weights, (size, size))
    denominator = F.adaptive_avg_pool2d(weights, (size, size))
    pooled = numerator / denominator.clamp_min(1e-8)
    pooled = torch.where(denominator > 0, pooled, torch.zeros_like(pooled))
    return pooled.flatten(2).transpose(1, 2), denominator.flatten(1) > 0


class SpatialAlignment(nn.Module):
    def __init__(self, visual_width=1152, spatial_width=2048, width=768, heads=12,
                 scene_branch=False, dropout=0.1):
        super().__init__()
        self.visual_width, self.spatial_width = visual_width, spatial_width
        self.local_visual_norm = nn.LayerNorm(visual_width)
        self.local_input_norm = nn.LayerNorm(spatial_width)
        self.local_query = nn.Linear(visual_width, width)
        self.local_input = nn.Linear(spatial_width, width)
        self.local_types = nn.Parameter(torch.zeros(2, width))
        nn.init.normal_(self.local_types, std=0.02)
        self.local_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.local_output = nn.Linear(width, visual_width)
        self.local_gate = nn.Parameter(torch.full((visual_width,), 1e-3))
        self.dropout = nn.Dropout(dropout)
        self.scene_branch = scene_branch
        if scene_branch:
            self.scene_input_norm = nn.LayerNorm(spatial_width)
            self.scene_input = nn.Linear(spatial_width, width)
            self.scene_types = nn.Parameter(torch.zeros(2, width))
            nn.init.normal_(self.scene_types, std=0.02)
            self.scene_blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
                for _ in range(2)
            ])
            self.scene_visual_norm = nn.LayerNorm(visual_width)
            self.scene_query = nn.Linear(visual_width, width)
            self.scene_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            self.scene_output = nn.Linear(width, visual_width)
            self.scene_gate = nn.Parameter(torch.full((visual_width,), 1e-3))

    def forward(self, visual, local, scene=None, visual_valid=None, local_valid=None, scene_valid=None):
        if visual.ndim != 3 or visual.shape[-1] != self.visual_width:
            raise ValueError("Invalid native visual tensor")
        batch = visual.shape[0]
        if local.shape != (batch, 197, self.spatial_width):
            raise ValueError("Local inputs must be 196 pooled patches followed by one camera token")
        if visual_valid is None:
            visual_valid = torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device)
        if local_valid is None:
            local_valid = torch.ones(local.shape[:2], dtype=torch.bool, device=local.device)
        if not visual_valid.any(1).all() or not local_valid.any(1).all():
            raise ValueError("Each example must contain valid visual and spatial tokens")
        local_type_ids = torch.zeros(197, dtype=torch.long, device=local.device)
        local_type_ids[-1] = 1
        keys = self.local_input(self.local_input_norm(local)) + self.local_types[local_type_ids]
        query = self.local_query(self.local_visual_norm(visual))
        context, _ = self.local_attention(query, keys, keys, key_padding_mask=~local_valid, need_weights=False)
        fused = visual + torch.tanh(self.local_gate) * self.dropout(self.local_output(context))
        if self.scene_branch:
            if scene is None or scene.shape != (batch, 17, self.spatial_width):
                raise ValueError("Scene inputs must be one camera and 16 register/pooled tokens")
            if scene_valid is None:
                scene_valid = torch.ones(scene.shape[:2], dtype=torch.bool, device=scene.device)
            if not scene_valid.any(1).all():
                raise ValueError("Each example must have at least one valid scene token")
            scene_type_ids = torch.ones(17, dtype=torch.long, device=scene.device)
            scene_type_ids[0] = 0
            scene_tokens = self.scene_input(self.scene_input_norm(scene)) + self.scene_types[scene_type_ids]
            for block in self.scene_blocks:
                scene_tokens = block(scene_tokens, src_key_padding_mask=~scene_valid)
            summary = (visual * visual_valid.unsqueeze(-1)).sum(1) / visual_valid.sum(1, keepdim=True)
            summary_query = self.scene_query(self.scene_visual_norm(summary)).unsqueeze(1)
            scene_context, _ = self.scene_attention(summary_query, scene_tokens, scene_tokens,
                                                     key_padding_mask=~scene_valid, need_weights=False)
            fused = fused + torch.tanh(self.scene_gate) * self.dropout(self.scene_output(scene_context))
        return torch.where(visual_valid.unsqueeze(-1), fused, visual)
