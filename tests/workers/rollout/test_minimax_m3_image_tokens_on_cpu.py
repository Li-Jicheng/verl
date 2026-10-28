# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MiniMax token transport must let vLLM insert image boundaries exactly once."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

_SOURCE = Path(__file__).resolve().parents[3] / "verl/workers/rollout/utils.py"
_FUNCTION = next(
    node
    for node in ast.parse(_SOURCE.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == "minimax_m3_vl_collapse_image_tokens"
)
_SCOPE = {}
exec(compile(ast.Module(body=[_FUNCTION], type_ignores=[]), str(_SOURCE), "exec"), _SCOPE)
minimax_m3_vl_collapse_image_tokens = _SCOPE["minimax_m3_vl_collapse_image_tokens"]


@pytest.fixture
def processor():
    vocab = {
        "]<]image[>[": 11,
        "]<]start of image[>[": 10,
        "]<]end of image[>[": 12,
    }
    return SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda: vocab))


@pytest.mark.parametrize(
    "prompt, image_count, expected",
    [
        ([1, 10, 11, 11, 11, 12, 2], 1, [1, 11, 2]),
        ([10, 11, 12, 10, 11, 11, 12], 2, [11, 11]),
        ([1, 11, 2], 1, [1, 11, 2]),
        ([10, 13, 13, 12, 10, 11, 12], 1, [10, 13, 13, 12, 11]),
        ([1, 2], 0, [1, 2]),
    ],
    ids=["expanded", "adjacent_images", "raw_placeholder", "preserve_video", "text_only"],
)
def test_restore_raw_image_placeholders_without_mutating_prompt(processor, prompt, image_count, expected):
    original = list(prompt)
    assert minimax_m3_vl_collapse_image_tokens(prompt, processor, image_count) == expected
    assert prompt == original


@pytest.mark.parametrize("prompt, image_count", [([10, 11, 11], 1), ([10, 11, 12], 2), ([11, 11], 1)])
def test_reject_malformed_or_mismatched_images(processor, prompt, image_count):
    with pytest.raises(ValueError):
        minimax_m3_vl_collapse_image_tokens(prompt, processor, image_count)
