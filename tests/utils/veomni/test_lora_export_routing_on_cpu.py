# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0.
"""Exercise actual engine export methods without importing accelerator backends."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "verl/workers/engine/veomni/transformer_impl.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
ENGINE = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "VeOmniEngine")
METHODS = {
    "get_per_tensor_param",
    "_get_hf_export_per_tensor_param",
    "set_rollout_weight_quantizer",
    "_quantize_for_rollout",
}
CHECKPOINT = next(
    n
    for n in ast.walk(ENGINE)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "FSDPCheckpointManager"
)
CALLBACK = next(k.value for k in CHECKPOINT.keywords if k.arg == "per_tensor_param_fn")


def make_engine(monkeypatch, model_type, *, lora=True):
    exports = []
    source_weight = torch.tensor([[1.25, -2.75]], dtype=torch.float32)

    def merged(model, device, *args):
        assert device == "cpu"
        exports.append(model.config.model_type)
        return iter([("weight", source_weight.clone())])

    package = ModuleType("_veomni_export_routing")
    package.__path__ = []
    m3 = ModuleType(package.__name__ + ".m3_lora")
    m3.merged_m3_lora_weights = merged
    dense = ModuleType("verl.utils.veomni.lora")
    dense.merged_dense_lora_weights = merged
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, m3.__name__, m3)
    monkeypatch.setitem(sys.modules, dense.__name__, dense)
    scope = {
        "__name__": package.__name__ + ".engine",
        "__package__": package.__name__,
        "get_device_id": lambda: "cpu",
        "parallel_state": SimpleNamespace(get_parallel_state=lambda: None),
        "get_checkpoint_tensor_converter": lambda model: SimpleNamespace(export_weights=lambda m: merged(m, "cpu")),
    }
    methods = [n for n in ENGINE.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    assert {n.name for n in methods} == METHODS
    cls = ast.ClassDef(name="Engine", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), scope)
    engine = scope["Engine"]()
    engine.module = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    engine._is_lora = lora
    engine._uses_fsdp2_cpu_offload_policy = True
    engine.set_rollout_weight_quantizer(None)
    return engine, source_weight, exports


@pytest.mark.parametrize("model_type", ["minimax_m3_vl", "qwen3"])
@pytest.mark.parametrize("quantized", [False, True])
def test_checkpoint_stays_float_and_rollout_quantizes_after_merge(monkeypatch, model_type, quantized):
    engine, source_weight, exports = make_engine(monkeypatch, model_type)
    calls = []

    def quantize(weights):
        values = dict(weights)
        torch.testing.assert_close(values["weight"], source_weight)
        calls.append("quantized")
        return iter([("weight", values["weight"].to(torch.int8))])

    engine.set_rollout_weight_quantizer(quantize if quantized else None)
    callback = eval(compile(ast.Expression(CALLBACK), str(SOURCE), "eval"), {"self": engine})
    checkpoint, checkpoint_peft = callback()
    torch.testing.assert_close(dict(checkpoint)["weight"], source_weight)
    assert checkpoint_peft is None and calls == []
    rollout, rollout_peft = engine.get_per_tensor_param()
    expected = source_weight.to(torch.int8) if quantized else source_weight
    torch.testing.assert_close(dict(rollout)["weight"], expected)
    assert rollout_peft is None
    assert calls == (["quantized"] if quantized else [])
    assert exports == [model_type, model_type]
    torch.testing.assert_close(source_weight, torch.tensor([[1.25, -2.75]]))


def test_non_lora_converter_retains_checkpoint_rollout_separation(monkeypatch):
    engine, source_weight, _ = make_engine(monkeypatch, "deepseek_v4", lora=False)
    engine.set_rollout_weight_quantizer(lambda weights: ((name, value.to(torch.int8)) for name, value in weights))
    checkpoint, _ = engine._get_hf_export_per_tensor_param()
    rollout, _ = engine.get_per_tensor_param()
    torch.testing.assert_close(dict(checkpoint)["weight"], source_weight)
    torch.testing.assert_close(dict(rollout)["weight"], source_weight.to(torch.int8))
