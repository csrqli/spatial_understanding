"""Pinned FLA kernels for Qwen3.5; explicit numerical checks against HF."""
import torch
import inspect


def configure(validate=False):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    report = {}
    if validate:
        torch.manual_seed(812)
        base = [torch.randn(2, 128, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        base += [-torch.rand(2,128,4,device="cuda"), torch.rand(2,128,4,device="cuda",dtype=torch.bfloat16)]
        left = [v.clone().requires_grad_() for v in base]
        right = [v.clone().requires_grad_() for v in base]
        # Transformers auto-dispatches its decorated function to installed FLA.
        # Unwrap every decorator so the reference is actually the torch algorithm.
        reference = inspect.unwrap(qwen.torch_chunk_gated_delta_rule)
        assert reference.__name__ == "torch_chunk_gated_delta_rule"
        a,_ = reference(*left[:3], g=left[3], beta=left[4], use_qk_l2norm_in_kernel=True)
        b,_ = chunk_gated_delta_rule(*right[:3], g=right[3], beta=right[4], use_qk_l2norm_in_kernel=True)
        upstream = torch.randn_like(a)
        (a*upstream).float().sum().backward()
        (b*upstream).float().sum().backward()
        report["forward_relative_l2"] = ((a.float()-b.float()).norm()/a.float().norm()).item()
        report["gradient_relative_l2"] = [((l.grad.float()-r.grad.float()).norm()/l.grad.float().norm()).item() for l,r in zip(left,right)]
        assert report["forward_relative_l2"] < .03, report
        assert max(report["gradient_relative_l2"]) < .05, report
    qwen.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    qwen.torch_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
    return report
