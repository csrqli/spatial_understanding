"""Identity-initialized local alignment and normalized per-token scene ablation."""
import torch
from torch import nn


class Alignment(nn.Module):
    def __init__(self, scene=False, width=512, dropout=.05):
        super().__init__()
        self.scene_branch=scene
        self.visual_norm=nn.LayerNorm(1152)
        self.local_norm=nn.LayerNorm(2048)
        self.query=nn.Linear(1152,width)
        self.local_input=nn.Linear(2048,width)
        self.local_types=nn.Parameter(torch.randn(2,width)*.02)
        self.local_attention=nn.MultiheadAttention(width,8,dropout=dropout,batch_first=True)
        self.local_output_norm=nn.LayerNorm(width)
        self.local_output=nn.Linear(width,1152)
        self.local_gate=nn.Parameter(torch.full((1152,),.1))
        self.dropout=nn.Dropout(dropout)
        nn.init.zeros_(self.local_output.weight);nn.init.zeros_(self.local_output.bias)
        if scene:
            self.scene_norm=nn.LayerNorm(2048)
            self.scene_input=nn.Linear(2048,width)
            self.scene_types=nn.Parameter(torch.randn(2,width)*.02)
            self.scene_block=nn.TransformerEncoderLayer(width,8,2*width,dropout=dropout,
                                                       activation='gelu',batch_first=True,norm_first=True)
            self.scene_output_norm=nn.LayerNorm(width)
            self.scene_attention=nn.MultiheadAttention(width,8,dropout=dropout,batch_first=True)
            self.scene_context_norm=nn.LayerNorm(width)
            self.scene_output=nn.Linear(width,1152)
            self.scene_gate=nn.Parameter(torch.full((1152,),.1))
            nn.init.zeros_(self.scene_output.weight);nn.init.zeros_(self.scene_output.bias)
        self.last_diagnostics={}

    def forward(self,visual,local,scene=None,visual_valid=None,local_valid=None,scene_valid=None):
        if visual_valid is None:visual_valid=torch.ones(visual.shape[:2],dtype=torch.bool,device=visual.device)
        if local_valid is None:local_valid=torch.ones(local.shape[:2],dtype=torch.bool,device=local.device)
        assert local.shape[1:]==(197,2048)
        ids=torch.zeros(197,dtype=torch.long,device=visual.device);ids[-1]=1
        q=self.query(self.visual_norm(visual))
        kv=self.local_input(self.local_norm(local))+self.local_types[ids]
        x,_=self.local_attention(q,kv,kv,key_padding_mask=~local_valid,need_weights=False)
        local_delta=self.local_output(self.local_output_norm(x))*self.local_gate.tanh()
        fused=visual+self.dropout(local_delta)
        scene_delta=None
        if self.scene_branch:
            assert scene is not None and scene.shape[1:]==(17,2048)
            if scene_valid is None:scene_valid=torch.ones(scene.shape[:2],dtype=torch.bool,device=scene.device)
            ids=torch.ones(17,dtype=torch.long,device=visual.device);ids[0]=0
            kv=self.scene_input(self.scene_norm(scene))+self.scene_types[ids]
            kv=self.scene_output_norm(self.scene_block(kv,src_key_padding_mask=~scene_valid))
            x,_=self.scene_attention(q,kv,kv,key_padding_mask=~scene_valid,need_weights=False)
            scene_delta=self.scene_output(self.scene_context_norm(x))*self.scene_gate.tanh()
            fused=fused+self.dropout(scene_delta)
        with torch.no_grad():
            denom=visual[visual_valid].float().square().mean().sqrt().clamp_min(1e-8)
            self.last_diagnostics={'local_relative_rms':float(local_delta[visual_valid].float().square().mean().sqrt()/denom),
                                   'local_gate_abs_mean':float(self.local_gate.tanh().abs().mean())}
            if scene_delta is not None:
                self.last_diagnostics.update(scene_relative_rms=float(scene_delta[visual_valid].float().square().mean().sqrt()/denom),
                                             scene_gate_abs_mean=float(self.scene_gate.tanh().abs().mean()))
        # Preserve the merger's input dtype as well as its values at identity.
        # Otherwise a FP32 gate promotes BF16 visual features, changing the
        # native LayerNorm rounding even when both residuals are exactly zero.
        return torch.where(visual_valid.unsqueeze(-1),fused,visual).to(visual.dtype)
