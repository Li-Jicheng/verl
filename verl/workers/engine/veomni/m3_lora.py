# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0.
"""Native MiniMax M3 adapters and non-mutating public-layout rollout export."""

import torch
from torch.distributed.tensor import DTensor

from .m3_weight_export import M3ExportSpec, iter_m3_param

_TEXT_TARGETS = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(shared_experts\.)?(gate_up_proj|down_proj))"
)
_EXPERT_TARGETS = [
    "model.language_model.layers.*.mlp.experts.gate_up_proj",
    "model.language_model.layers.*.mlp.experts.down_proj",
]


def configure_m3_lora(config):
    """Make all-linear text-only, including experts; preserve explicit targets."""
    changes = {}
    if config.target_modules == "all-linear":
        changes["target_modules"] = _TEXT_TARGETS
        if not config.target_parameters:
            changes["target_parameters"] = list(_EXPERT_TARGETS)
    if changes.get("target_parameters", config.target_parameters) and config.moe_mode is None:
        changes["moe_mode"] = "independent"
    return config.replace(**changes)


def prepare_m3_lora_dtypes(model):
    """Before sharding, retain FP32 adapters/router and BF16 frozen matrices.

    Construct the model in FP32 first so router and correction-bias precision is
    retained. The companion VeOmni preserve_frozen_dtype option must also be set
    to prevent build_parallelize_model from undoing this storage policy.
    """
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        keep_fp32 = parameter.requires_grad or name == "mlp.gate.weight" or name.endswith(".mlp.gate.weight")
        parameter.data = parameter.data.to(torch.float32 if keep_fp32 else torch.bfloat16)
    for name, buffer in model.named_buffers():
        if name.endswith(".e_score_correction_bias") and buffer.is_floating_point():
            buffer.data = buffer.data.float()


def _merge_projection(base, a, b, scale):
    """Merge dense, shared, or per-expert B@A with bounded FP32 row storage."""
    if base.ndim not in (2, 3) or a.ndim != b.ndim or a.ndim not in (2, 3):
        raise ValueError("Unsupported MiniMax LoRA projection layout")
    if base.ndim == 2 and a.ndim != 2:
        raise ValueError("Dense MiniMax projection requires 2D adapters")
    if a.shape[-2] != b.shape[-1] or (b.shape[-2], a.shape[-1]) != tuple(base.shape[-2:]):
        raise ValueError("MiniMax LoRA adapter and base shapes disagree")
    if a.ndim == 3 and (a.shape[0] != base.shape[0] or b.shape[0] != base.shape[0]):
        raise ValueError("MiniMax expert adapter count disagrees with local base")
    result = base.clone()
    if base.ndim == 2:
        return torch.addmm(base.float(), b.float(), a.float(), alpha=scale).to(base.dtype)
    for expert in range(base.shape[0]):
        ae, be = (a[expert], b[expert]) if a.ndim == 3 else (a, b)
        result[expert] = torch.addmm(base[expert].float(), be.float(), ae.float(), alpha=scale).to(base.dtype)
    return result


def merged_m3_lora_weights(model, device, parallel_state):
    """Yield merged public weights; never merge into actor/optimizer parameters.

    All training ranks must exhaust this iterator in identical order. FSDP
    gathers a local expert stack and its adapters; the export helper streams
    expert rows across EP ranks. This is full weight sync, not adapter-only sync.
    """
    from veomni.lora import VeOmniLoraModel
    from veomni.lora.layers import LoraLinear
    from veomni.lora.moe_layers import is_lora_moe_experts

    if not isinstance(model, VeOmniLoraModel):
        raise TypeError("Expected a VeOmni-native LoRA model")
    spec = M3ExportSpec.from_config(model.config)
    dense, expert_weights = {}, {}
    for name, layer in model.named_modules():
        if isinstance(layer, LoraLinear):
            if layer.merged:
                raise ValueError("MiniMax export requires unmerged training adapters")
            dense[name] = layer
        elif is_lora_moe_experts(layer):
            for projection in ("gate_up_proj", "down_proj"):
                expert_weights[f"{name}.{projection}"] = (layer, projection)

    def full(tensor):
        tensor = tensor.detach().to(device, non_blocking=True)
        return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor

    with torch.no_grad():
        for name, parameter in model.state_dict().items():
            if any(part in {"lora_A", "lora_B", "lora_scaling"} for part in name.split(".")):
                continue
            value = full(parameter)
            stem = name.removesuffix(".base_layer.weight")
            if stem in dense:
                layer = dense[stem]
                adapter = layer.active_adapter
                value = _merge_projection(
                    value,
                    full(layer.lora_A[adapter].weight),
                    full(layer.lora_B[adapter].weight),
                    layer.scaling[adapter],
                )
                name = stem + ".weight"
            elif stem in expert_weights:
                layer, projection = expert_weights[stem]
                if projection == "gate_up_proj":
                    gate, up = value.chunk(2, dim=1)
                    value = torch.cat(
                        [
                            _merge_projection(
                                base,
                                full(layer.get_lora_A_weight(logical)),
                                full(layer.get_lora_B_weight(logical)),
                                layer._lora_scale_value,
                            )
                            for logical, base in (("gate_proj", gate), ("up_proj", up))
                        ],
                        dim=1,
                    )
                else:
                    value = _merge_projection(
                        value,
                        full(layer.get_lora_A_weight("down_proj")),
                        full(layer.get_lora_B_weight("down_proj")),
                        layer._lora_scale_value,
                    )
                name = stem
            # Frozen dense biases keep their value but lose the native wrapper prefix.
            if name.endswith(".base_layer.bias"):
                bias_stem = name.removesuffix(".base_layer.bias")
                if bias_stem in dense:
                    name = bias_stem + ".bias"
            name = name.removeprefix("base_model.model.")
            yield from iter_m3_param(name, value, spec, parallel_state)
