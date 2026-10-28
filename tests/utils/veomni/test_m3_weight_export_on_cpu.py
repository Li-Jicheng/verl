# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0.
"""Check public MiniMax export values, model dimensions and EP row ownership."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_PATH = Path(__file__).resolve().parents[3] / "verl/workers/engine/veomni/m3_weight_export.py"
_SPEC = importlib.util.spec_from_file_location("m3_weight_export_contract", _PATH)
m3 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = m3
_SPEC.loader.exec_module(m3)


def config(**overrides):
    text = dict(
        num_hidden_layers=2,
        num_local_experts=4,
        hidden_size=3,
        intermediate_size=2,
        dense_intermediate_size=5,
        shared_intermediate_size=2,
        moe_layer_freq=[0, 1],
    )
    text.update(overrides)
    return dict(text_config=text, vision_config=dict(num_hidden_layers=2))


def test_expert_gate_up_exports_public_weights_with_correct_expert_ids():
    spec = m3.M3ExportSpec.from_config(config())
    source = torch.arange(24.0).reshape(2, 4, 3)
    before = source.clone()
    out = dict(
        m3.export_m3_tensor("model.language_model.layers.1.mlp.experts.gate_up_proj", source, spec, expert_id_base=2)
    )
    for row, expert in enumerate((2, 3)):
        prefix = f"language_model.model.layers.1.block_sparse_moe.experts.{expert}."
        assert torch.equal(out[prefix + "w1.weight"], source[row, :2])
        assert torch.equal(out[prefix + "w3.weight"], source[row, 2:])
    assert torch.equal(source, before)


@pytest.mark.parametrize(
    "name,target,shape",
    [
        ("lm_head.weight", "language_model.lm_head.weight", (7, 3)),
        ("model.language_model.embed_tokens.weight", "language_model.model.embed_tokens.weight", (7, 3)),
        (
            "model.language_model.layers.1.self_attn.indexer.k_norm.weight",
            "language_model.model.layers.1.self_attn.index_k_norm.weight",
            (3,),
        ),
        (
            "model.language_model.layers.1.mlp.experts.down_proj",
            "language_model.model.layers.1.block_sparse_moe.experts.0.w2.weight",
            (4, 3, 2),
        ),
        (
            "model.vision_tower.embeddings.proj.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            (3, 3),
        ),
        (
            "model.vision_tower.layers.1.self_attn.q_proj.weight",
            "vision_tower.vision_model.encoder.layers.1.self_attn.q_proj.weight",
            (3, 3),
        ),
        ("model.multi_modal_projector.merge_linear_1.weight", "patch_merge_mlp.linear_1.weight", (3, 3)),
    ],
)
def test_plain_vision_projector_and_expert_mappings(name, target, shape):
    source = torch.randn(shape)
    out = dict(m3.export_m3_tensor(name, source, m3.M3ExportSpec.from_config(config())))
    assert target in out
    expected = source[0] if "experts.down_proj" in name else source
    assert torch.equal(out[target], expected)


@pytest.mark.parametrize("layer,shared,width", [(0, False, 5), (1, True, 2)])
def test_dense_and_shared_dimensions_are_independent(layer, shared, width):
    name = f"model.language_model.layers.{layer}.mlp." + ("shared_experts." if shared else "")
    source = torch.randn(2 * width, 3)
    out = list(m3.export_m3_tensor(name + "gate_up_proj.weight", source, m3.M3ExportSpec.from_config(config())))
    assert len(out) == 2
    assert out[0][0].endswith("gate_proj.weight")
    assert out[1][0].endswith("up_proj.weight")
    assert torch.equal(torch.cat([pair[1] for pair in out]), source)


@pytest.mark.parametrize(
    "name,shape",
    [
        ("model.language_model.layers.2.input_layernorm.weight", (3,)),
        ("model.language_model.layers.0.mlp.experts.down_proj", (4, 3, 2)),
        ("model.language_model.layers.1.mlp.gate.weight", (3, 3)),
        ("model.language_model.layers.1.mlp.experts.gate_up_proj", (4, 3, 3)),
        ("model.vision_tower.layers.2.self_attn.q_proj.weight", (3, 3)),
        ("unknown.weight", (3,)),
    ],
)
def test_invalid_layouts_fail_before_sending_weights(name, shape):
    with pytest.raises(ValueError):
        list(m3.export_m3_tensor(name, torch.zeros(shape), m3.M3ExportSpec.from_config(config())))


@pytest.mark.parametrize(
    "overrides",
    [
        dict(num_local_experts=0),
        dict(dense_intermediate_size=0),
        dict(moe_layer_freq=[1]),
        dict(mlp_layer_types=["unknown", "sparse"]),
    ],
)
def test_invalid_model_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        m3.M3ExportSpec.from_config(config(**overrides))


def test_quantized_configuration_and_integer_state_are_rejected():
    cfg = config()
    cfg["quantization_config"] = dict(quant_method="fp8")
    with pytest.raises(ValueError, match="unquantized"):
        m3.M3ExportSpec.from_config(cfg)
    with pytest.raises(ValueError, match="non-floating"):
        list(
            m3.export_m3_tensor(
                "lm_head.weight", torch.zeros(3, dtype=torch.int32), m3.M3ExportSpec.from_config(config())
            )
        )


def _ep_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        spec = m3.M3ExportSpec.from_config(config())
        state = SimpleNamespace(ep_enabled=True, ep_size=2, ep_rank=rank, ep_group=dist.group.WORLD)
        source = torch.stack([torch.full((4, 3), float(2 * rank + i)) for i in range(2)])
        before = source.clone()
        values = dict(m3.iter_m3_param("model.language_model.layers.1.mlp.experts.gate_up_proj", source, spec, state))
        for expert in range(4):
            prefix = f"language_model.model.layers.1.block_sparse_moe.experts.{expert}."
            for projection in ("w1", "w3"):
                assert torch.equal(values[prefix + projection + ".weight"], torch.full((2, 3), float(expert)))
        assert torch.equal(source, before)
    finally:
        dist.destroy_process_group()


def test_two_rank_ep_export_keeps_all_yielded_rows_alive(tmp_path):
    if not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    rendezvous = (tmp_path / "m3_ep_store").as_uri()
    mp.spawn(_ep_worker, args=(rendezvous,), nprocs=2, join=True)
