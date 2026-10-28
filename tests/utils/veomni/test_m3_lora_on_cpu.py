# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0.
"""CPU merge/export contracts. Does not validate a sharded accelerator model."""

import importlib
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = "_m3_lora_export_contract"
package = types.ModuleType(_PACKAGE)
package.__path__ = [str(_ROOT / "verl/workers/engine/veomni")]
sys.modules[_PACKAGE] = package
m3 = importlib.import_module(_PACKAGE + ".m3_lora")


@dataclass
class Config:
    target_modules: object = "all-linear"
    target_parameters: object = None
    moe_mode: object = None

    def replace(self, **changes):
        return replace(self, **changes)


def test_default_targets_are_text_only_and_include_independent_experts():
    import re

    original = Config()
    config = m3.configure_m3_lora(original)
    assert config.moe_mode == "independent" and len(config.target_parameters) == 2
    assert re.fullmatch(config.target_modules, "model.language_model.layers.4.self_attn.q_proj")
    assert not re.fullmatch(config.target_modules, "model.language_model.layers.4.self_attn.indexer.q_proj")
    assert not re.fullmatch(config.target_modules, "model.vision_tower.layers.0.self_attn.q_proj")
    assert not re.fullmatch(config.target_modules, "lm_head")
    assert original.target_modules == "all-linear" and original.target_parameters is None


def test_explicit_dense_targets_are_preserved():
    config = m3.configure_m3_lora(Config(target_modules=["q_proj"]))
    assert config.target_modules == ["q_proj"] and config.target_parameters is None


def test_explicit_shared_expert_mode_is_preserved():
    config = m3.configure_m3_lora(Config(target_parameters=["*.experts.down_proj"], moe_mode="shared"))
    assert config.moe_mode == "shared" and config.target_parameters == ["*.experts.down_proj"]


@pytest.mark.parametrize("layout", ["dense", "shared", "independent"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_projection_merge_matches_matmul_without_mutating_inputs(layout, dtype):
    torch.manual_seed(68)
    base = torch.randn((4, 3) if layout == "dense" else (2, 4, 3), dtype=dtype)
    prefix = (2,) if layout == "independent" else ()
    a = torch.randn((*prefix, 2, 3), dtype=dtype)
    b = torch.randn((*prefix, 4, 2), dtype=dtype)
    original = [x.clone() for x in (base, a, b)]
    expected = (base.float() + (b.float() @ a.float()) * 0.5).to(dtype)
    actual = m3._merge_projection(base, a, b, 0.5)
    torch.testing.assert_close(
        actual, expected, rtol=1e-6 if dtype == torch.float32 else 0, atol=1e-6 if dtype == torch.float32 else 0
    )
    assert actual.dtype == dtype
    for current, before in zip((base, a, b), original, strict=True):
        assert torch.equal(current, before)


@pytest.mark.parametrize(
    "shapes", [((4, 3), (2, 2, 3), (2, 4, 2)), ((2, 4, 3), (3, 2, 3), (3, 4, 2)), ((4, 3), (2, 5), (4, 2))]
)
def test_invalid_adapter_shapes_fail_before_export(shapes):
    with pytest.raises(ValueError):
        m3._merge_projection(*(torch.ones(shape) for shape in shapes), 1.0)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_m3_storage_policy_keeps_router_precision_and_parameter_identity(device):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.ones(3, device=device), requires_grad=False)
            self.adapter = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16, device=device))
            self.mlp = torch.nn.Module()
            self.mlp.gate = torch.nn.Linear(3, 2, bias=False, device=device)
            self.mlp.gate.weight.requires_grad = False
            self.mlp.register_buffer("e_score_correction_bias", torch.ones(2, device=device))

    model = Model()
    before = list(model.parameters())
    m3.prepare_m3_lora_dtypes(model)
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
    assert model.base.dtype == torch.bfloat16
    assert model.adapter.dtype == model.mlp.gate.weight.dtype == torch.float32
    assert model.mlp.e_score_correction_bias.dtype == torch.float32


@pytest.mark.parametrize("layout", ["shared", "independent"])
def test_merged_export_maps_all_expert_halves_and_omits_adapter_state(monkeypatch, layout):
    """Exercise the exporter against native wrapper's public accessor protocol."""

    class Native:
        pass

    class Linear:
        pass

    class Expert:
        pass

    for name in ("veomni", "veomni.lora", "veomni.lora.layers", "veomni.lora.moe_layers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["veomni.lora"].VeOmniLoraModel = Native
    sys.modules["veomni.lora.layers"].LoraLinear = Linear
    sys.modules["veomni.lora.moe_layers"].is_lora_moe_experts = lambda layer: isinstance(layer, Expert)
    torch.manual_seed(73)
    expert = Expert()
    prefix = (2,) if layout == "independent" else ()
    adapters = {}
    for logical, (out_size, in_size) in {"gate_proj": (2, 3), "up_proj": (2, 3), "down_proj": (3, 2)}.items():
        adapters[logical] = (torch.randn((*prefix, 1, in_size)), torch.randn((*prefix, out_size, 1)))
    expert.get_lora_A_weight = lambda name: adapters[name][0]
    expert.get_lora_B_weight = lambda name: adapters[name][1]
    expert._lora_scale_value = 0.5
    dense = Linear()
    dense.merged, dense.active_adapter = False, "custom"
    dense.lora_A = {"custom": SimpleNamespace(weight=torch.randn(1, 3))}
    dense.lora_B = {"custom": SimpleNamespace(weight=torch.randn(3, 1))}
    dense.scaling = {"custom": 0.7}
    ep_name = "base_model.model.model.language_model.layers.0.mlp.experts"
    dense_name = "base_model.model.model.language_model.layers.0.self_attn.q_proj"
    state = {
        ep_name + ".gate_up_proj.base_layer.weight": torch.randn(2, 4, 3),
        ep_name + ".down_proj.base_layer.weight": torch.randn(2, 3, 2),
        ep_name + ".gate_proj.lora_A.default.weight": adapters["gate_proj"][0],
        ep_name + ".lora_scaling": torch.tensor(0.5),
        dense_name + ".base_layer.weight": torch.randn(3, 3),
    }
    before = {name: value.clone() for name, value in state.items()}
    model = Native()
    model.config = dict(
        text_config=dict(
            num_hidden_layers=1, num_local_experts=2, hidden_size=3, intermediate_size=2, moe_layer_freq=[1]
        )
    )
    model.named_modules = lambda: [(ep_name, expert), (dense_name, dense)]
    model.state_dict = lambda: state
    ps = SimpleNamespace(ep_enabled=False, ep_size=1, ep_rank=0)
    out = dict(m3.merged_m3_lora_weights(model, "cpu", ps))
    assert len(out) == 7 and all("lora_" not in name for name in out)
    gu = before[ep_name + ".gate_up_proj.base_layer.weight"]
    dn = before[ep_name + ".down_proj.base_layer.weight"]
    for e in range(2):
        for logical, public, base in (
            ("gate_proj", "w1", gu[e, :2]),
            ("up_proj", "w3", gu[e, 2:]),
            ("down_proj", "w2", dn[e]),
        ):
            a, b = adapters[logical]
            if layout == "independent":
                a, b = a[e], b[e]
            expected = base + 0.5 * b @ a
            key = f"language_model.model.layers.0.block_sparse_moe.experts.{e}.{public}.weight"
            torch.testing.assert_close(out[key], expected)
    expected_dense = (
        before[dense_name + ".base_layer.weight"] + 0.7 * dense.lora_B["custom"].weight @ dense.lora_A["custom"].weight
    )
    torch.testing.assert_close(out["language_model.model.layers.0.self_attn.q_proj.weight"], expected_dense)
    for name in state:
        assert torch.equal(state[name], before[name])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "target,public",
    [
        ("model.multi_modal_projector.linear_1", "multi_modal_projector.linear_1"),
        (
            "model.vision_tower.layers.0.self_attn.q_proj",
            "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        ),
    ],
)
def test_explicit_dense_targets_export_frozen_bias_without_wrapper_names(monkeypatch, dtype, target, public):
    """Use real module state keys; isolate native package startup, not the exporter."""

    class Dense(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(3, 3, bias=True, dtype=dtype)
            self.base_layer.requires_grad_(False)
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 1, bias=False, dtype=dtype)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 3, bias=False, dtype=dtype)})
            self.scaling = {"default": 0.5}
            self.active_adapter, self.merged = "default", False

    class Native(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = torch.nn.Module()
            self.base_model.model = torch.nn.Module()
            parent = self.base_model.model
            for component in target.split(".")[:-1]:
                child = torch.nn.Module()
                parent.add_module(component, child)
                parent = child
            parent.add_module(target.split(".")[-1], Dense())
            self.config = dict(
                text_config=dict(
                    num_hidden_layers=1, num_local_experts=2, hidden_size=3, intermediate_size=2, moe_layer_freq=[1]
                ),
                vision_config=dict(num_hidden_layers=1),
            )

    for name in ("veomni", "veomni.lora", "veomni.lora.layers", "veomni.lora.moe_layers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["veomni.lora"].VeOmniLoraModel = Native
    sys.modules["veomni.lora.layers"].LoraLinear = Dense
    sys.modules["veomni.lora.moe_layers"].is_lora_moe_experts = lambda layer: False
    torch.manual_seed(1702)
    model = Native()
    layer = model.get_submodule("base_model.model." + target)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    expected_weight = (
        layer.base_layer.weight.float()
        + 0.5 * layer.lora_B["default"].weight.float() @ layer.lora_A["default"].weight.float()
    ).to(dtype)
    ps = SimpleNamespace(ep_enabled=False, ep_size=1, ep_rank=0)
    exported = dict(m3.merged_m3_lora_weights(model, "cpu", ps))
    assert set(exported) == {public + ".weight", public + ".bias"}
    assert torch.equal(exported[public + ".bias"], layer.base_layer.bias)
    assert exported[public + ".bias"].dtype == dtype
    torch.testing.assert_close(exported[public + ".weight"], expected_weight)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
