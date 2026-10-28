# SPDX-License-Identifier: Apache-2.0
import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SOURCE_PATH = "verl/workers/engine/fsdp/transformer_impl.py"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / SOURCE_PATH).is_file())
TREE = ast.parse((ROOT / SOURCE_PATH).read_text(encoding="utf-8"))

NODE = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_scale_logits_by_temperature")
SCOPE = {"os": os, "torch": torch}
exec(compile(ast.Module(body=[NODE], type_ignores=[]), SOURCE_PATH, "exec"), SCOPE)
scale = SCOPE["_scale_logits_by_temperature"]


class TestFP32Logits(unittest.TestCase):
    def test_default_unit_temperature_is_identity(self):
        with patch.dict(os.environ, {}, clear=True):
            x = torch.randn(2, 17).bfloat16()
            self.assertIs(scale(x, 1.0, is_unit_temperature=True), x)

    def test_disabled_retains_original_dtype_and_result(self):
        x = torch.linspace(-3, 3, 34).reshape(2, 17).bfloat16()
        t = torch.tensor([[0.7], [1.3]])
        for flag in ("0", "false"):
            with patch.dict(os.environ, {"VERL_POLICY_FP32_LOGITS": flag}):
                actual = scale(x, t, is_unit_temperature=False)
                self.assertEqual(actual.dtype, x.dtype)
                self.assertTrue(torch.equal(actual, x / t.to(x.dtype)))

    def test_upcast_precedes_temperature_and_preserves_inputs(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), patch.dict(os.environ, {"VERL_POLICY_FP32_LOGITS": "1"}):
                x = torch.linspace(-3, 3, 34).reshape(2, 17).to(dtype)
                t = torch.tensor([[0.7], [1.3]])
                xb, tb = x.clone(), t.clone()
                actual = scale(x, t, is_unit_temperature=False)
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, x.float() / t, rtol=0, atol=0)
                self.assertFalse(torch.equal(actual, (x / t.to(dtype)).float()))
                self.assertTrue(torch.equal(x, xb))
                self.assertTrue(torch.equal(t, tb))

    def test_opt_in_unit_temperature(self):
        with patch.dict(os.environ, {"VERL_POLICY_FP32_LOGITS": "1"}):
            x = torch.randn(2, 17).bfloat16()
            actual = scale(x, 1.0, is_unit_temperature=True)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.equal(actual, x.float()))
            xf = x.float()
            self.assertIs(scale(xf, 1.0, is_unit_temperature=True), xf)

    def test_gradient_and_clamp(self):
        with patch.dict(os.environ, {"VERL_POLICY_FP32_LOGITS": "1"}):
            x = torch.randn(2, 17).bfloat16().requires_grad_()
            t = torch.tensor([[0.7], [1.3]])
            scale(x, t, is_unit_temperature=False).sum().backward()
            expected = (torch.ones(2, 17) / t).bfloat16()
            self.assertTrue(torch.equal(x.grad, expected))
            result = scale(torch.ones(1, 1).bfloat16(), torch.zeros(1, 1), is_unit_temperature=False)
            self.assertTrue(torch.isfinite(result).all())
            torch.testing.assert_close(result, torch.full((1, 1), 1e8))


if __name__ == "__main__":
    unittest.main()
