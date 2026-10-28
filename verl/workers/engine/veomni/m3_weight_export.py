# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0.
"""Export VeOmni MiniMax M3 tensors in the public checkpoint layout.

The inverse of VeOmni's MiniMax public-layout checkpoint import converter.
No layer count, expert count, hidden size, or vision depth is hard-coded.
Only full floating-point weight transfer is supported; this is not a quantizer.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.distributed as dist


def _get(config, name, default=None):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


@dataclass(frozen=True)
class M3ExportSpec:
    layers: int
    experts: int
    hidden: int
    intermediate: int
    dense_intermediate: int
    shared_intermediate: int
    moe_layers: tuple[bool, ...]
    vision_layers: int

    @classmethod
    def from_config(cls, config):
        text = _get(config, "text_config", config)
        if _get(config, "quantization_config") or _get(text, "quantization_config"):
            raise ValueError("M3 RL export currently requires an unquantized checkpoint")
        layers = int(_get(text, "num_hidden_layers"))
        kinds = _get(text, "mlp_layer_types")
        if kinds is not None:
            if any(k not in ("sparse", "dense") for k in kinds):
                raise ValueError("Unknown M3 mlp_layer_types")
            moe = tuple(k == "sparse" for k in kinds)
        else:
            frequency = _get(text, "moe_layer_freq")
            moe = tuple(bool(x) for x in frequency) if frequency is not None else (True,) * layers
        if len(moe) != layers:
            raise ValueError("M3 MoE layer schedule length must equal num_hidden_layers")
        intermediate = int(_get(text, "intermediate_size"))
        result = cls(
            layers=layers,
            experts=int(_get(text, "num_local_experts", _get(text, "num_experts"))),
            hidden=int(_get(text, "hidden_size")),
            intermediate=intermediate,
            dense_intermediate=int(_get(text, "dense_intermediate_size", intermediate)),
            shared_intermediate=int(_get(text, "shared_intermediate_size", intermediate)),
            moe_layers=moe,
            vision_layers=int(_get(_get(config, "vision_config", {}), "num_hidden_layers", 0)),
        )
        if (
            min(
                result.layers,
                result.experts,
                result.hidden,
                result.intermediate,
                result.dense_intermediate,
                result.shared_intermediate,
            )
            < 1
        ):
            raise ValueError("M3 dimensions must be positive")
        if result.vision_layers < 0:
            raise ValueError("M3 vision depth must be nonnegative")
        return result


_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_EXPERT = re.compile(r"^mlp\.experts\.(gate_up_proj|down_proj)$")


def _check_shape(name, tensor, expected):
    if tuple(tensor.shape) != tuple(expected):
        raise ValueError(f"M3 export shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(expected)}")


def is_m3_expert(name):
    match = _LAYER.fullmatch(name)
    return match is not None and _EXPERT.fullmatch(match[2]) is not None


def export_m3_tensor(name, tensor, spec, *, expert_id_base=0):
    """Yield public-checkpoint names and tensors; preserve dtype and orientation."""
    if not tensor.is_floating_point():
        raise ValueError(f"Unsupported non-floating M3 state tensor: {name}")
    if name == "lm_head.weight":
        yield "language_model.lm_head.weight", tensor
        return
    if name in ("model.language_model.embed_tokens.weight", "model.language_model.norm.weight"):
        yield name.replace("model.language_model.", "language_model.model.", 1), tensor
        return

    match = _LAYER.fullmatch(name)
    if match:
        layer, suffix = int(match[1]), match[2]
        if layer >= spec.layers:
            raise ValueError(f"M3 layer {layer} is outside config depth {spec.layers}")
        prefix = f"language_model.model.layers.{layer}."
        expert = _EXPERT.fullmatch(suffix)
        if expert:
            if not spec.moe_layers[layer]:
                raise ValueError(f"Expert tensor in configured dense layer: {name}")
            count = tensor.shape[0]
            if expert_id_base < 0 or expert_id_base + count > spec.experts:
                raise ValueError(f"M3 expert range exceeds configured count: {name}")
            dims = (
                (2 * spec.intermediate, spec.hidden)
                if expert[1] == "gate_up_proj"
                else (spec.hidden, spec.intermediate)
            )
            _check_shape(name, tensor, (count, *dims))
            for local_id, row in enumerate(tensor):
                expert_prefix = f"{prefix}block_sparse_moe.experts.{expert_id_base + local_id}."
                if expert[1] == "gate_up_proj":
                    gate, up = row.chunk(2, dim=0)
                    yield expert_prefix + "w1.weight", gate.contiguous()
                    yield expert_prefix + "w3.weight", up.contiguous()
                else:
                    yield expert_prefix + "w2.weight", row.contiguous()
            return
        if suffix in ("mlp.gate.weight", "mlp.gate.e_score_correction_bias"):
            if not spec.moe_layers[layer]:
                raise ValueError(f"Router tensor in configured dense layer: {name}")
            out = (
                "block_sparse_moe.gate.weight"
                if suffix.endswith(".weight")
                else ("block_sparse_moe.e_score_correction_bias")
            )
            expected = (spec.experts, spec.hidden) if suffix.endswith(".weight") else (spec.experts,)
            _check_shape(name, tensor, expected)
            yield prefix + out, tensor
            return
        dense = re.fullmatch(r"mlp\.(shared_experts\.)?(gate_up_proj|down_proj)\.weight", suffix)
        if dense:
            shared = dense[1] is not None
            if shared != spec.moe_layers[layer]:
                raise ValueError(f"MLP layout does not match M3 config: {name}")
            intermediate = spec.shared_intermediate if shared else spec.dense_intermediate
            out = "block_sparse_moe.shared_experts." if shared else "mlp."
            if dense[2] == "gate_up_proj":
                _check_shape(name, tensor, (2 * intermediate, spec.hidden))
                gate, up = tensor.chunk(2, dim=0)
                yield prefix + out + "gate_proj.weight", gate.contiguous()
                yield prefix + out + "up_proj.weight", up.contiguous()
            else:
                _check_shape(name, tensor, (spec.hidden, intermediate))
                yield prefix + out + "down_proj.weight", tensor
            return
        indexer = re.fullmatch(r"self_attn\.indexer\.([qk])_(proj|norm)\.weight", suffix)
        if indexer:
            yield prefix + f"self_attn.index_{indexer[1]}_{indexer[2]}.weight", tensor
            return
        if re.fullmatch(
            r"(?:self_attn\.(?:[qkvo]_proj|[qk]_norm)|input_layernorm|post_attention_layernorm)\.(?:weight|bias)",
            suffix,
        ):
            yield prefix + suffix, tensor
            return
        raise ValueError(f"Unmapped M3 language tensor: {name}")

    if name.startswith("model.vision_tower."):
        suffix = name.removeprefix("model.vision_tower.")
        if suffix.startswith("layers."):
            layer = int(suffix.split(".")[1])
            if layer >= spec.vision_layers:
                raise ValueError(f"M3 vision layer {layer} exceeds configured depth")
            suffix = "encoder." + suffix
        elif suffix == "embeddings.proj.weight":
            suffix = "embeddings.patch_embedding.weight"
        elif suffix not in ("pre_layrnorm.weight", "pre_layrnorm.bias"):
            raise ValueError(f"Unmapped M3 vision tensor: {name}")
        yield "vision_tower.vision_model." + suffix, tensor
        return
    projector = re.fullmatch(r"model\.multi_modal_projector\.(merge_)?linear_([12])\.(weight|bias)", name)
    if projector:
        prefix = "patch_merge_mlp" if projector[1] else "multi_modal_projector"
        yield f"{prefix}.linear_{projector[2]}.{projector[3]}", tensor
        return
    raise ValueError(f"Unmapped M3 state tensor (MTP/quantized/other layout is unsupported): {name}")


def iter_m3_param(name, tensor, spec, parallel_state):
    """After FSDP gather, stream EP rows without gathering all experts at once.

    A fresh receive row is used each time so previously yielded tensor views are
    never overwritten by a later broadcast. FSDP still materializes one local
    expert-stack parameter; sufficient memory/EP sharding is required for it.
    """
    if not is_m3_expert(name):
        yield from export_m3_tensor(name, tensor, spec)
        return
    ep_size = parallel_state.ep_size if parallel_state.ep_enabled else 1
    if ep_size < 1 or not 0 <= parallel_state.ep_rank < ep_size:
        raise ValueError("Invalid M3 expert-parallel size or rank")
    if spec.experts % ep_size:
        raise ValueError("M3 expert count must divide evenly across training EP ranks")
    count = spec.experts // ep_size
    if tensor.ndim != 3 or tensor.shape[0] != count:
        raise ValueError(f"Expected {count} local M3 experts after FSDP gather, got {tuple(tensor.shape)}")
    if ep_size == 1:
        yield from export_m3_tensor(name, tensor, spec)
        return
    for source in range(ep_size):
        for row_id in range(count):
            row = tensor[row_id].contiguous() if parallel_state.ep_rank == source else torch.empty_like(tensor[0])
            dist.broadcast(row, group_src=source, group=parallel_state.ep_group)
            yield from export_m3_tensor(name, row.unsqueeze(0), spec, expert_id_base=source * count + row_id)
