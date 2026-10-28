MiniMax M3 with VeOmni and vLLM
==============================

This integration exports the unquantized ``minimax_m3_vl`` model from VeOmni's
training layout into the public checkpoint layout consumed by vLLM. Dimensions
and dense/MoE schedules come from the model config. It normalizes image
placeholders before vLLM processor expansion and checks the expanded prompt.

Dependencies
------------

The model and base SwiGLU-OAI operators currently require `VeOmni PR #1315
<https://github.com/ByteDance-Seed/VeOmni/pull/1315>`_, inspected at
``3fc30d7f306aa086e16859622b0369eab454f8df``. This is an unmerged source
dependency, not a released PyPI version. The verl integration is based on
``4d3ed12a7617c6b78ee0ac8f27dd7848bb0d11ee``. An exact combined VeOmni commit
must be published and recorded before this integration can be installed with a
frozen dependency lock; the existing default VeOmni pin is insufficient.

Use a vLLM/vllm-ascend build with matching MiniMax M3 public-layout support.
Record its commit, PyTorch, torch-npu and CANN versions as well as Python
package versions. A conda package list alone does not identify editable source
copies, import paths or native runtime libraries. No particular current
vLLM/vllm-ascend pair is newly hardware-validated by this change.

Usage
-----

``examples/grpo_trainer/run_minimax_m3_veomni.sh`` is an Ascend integration
template. Supply ``MODEL_PATH``, ``TRAIN_FILES``, ``VAL_FILES``, ``NNODES``,
``NPUS_PER_NODE``, ``TRAIN_EP``, ``TRAIN_SP``, ``ROLLOUT_TP`` and ``ROLLOUT_EP``
for a prepared Ray cluster. The script uses full weight synchronization and
eager rollout. Extra Hydra overrides are accepted at the end. It does not
provision machines or install dependencies, and the new template has only been
syntax-checked; size its batch and topology for the complete model.

Validation boundaries
---------------------

CPU tests cover public tensor names/shapes, image tokens and two-process Gloo
expert-row export. These tests need no model weights or accelerators. They do
not establish NPU kernel behavior, full-model memory use or numerical training
equivalence on the extracted source.

Historical full-size M3.1-Flash evidence used 60 layers and 128 experts on
16 machines/256 NPUs: training EP32/SP1 and rollout EP64, with DSpark disabled.
It completed two actor updates and synchronizations. Rewards were 0.6476563
and 0.7145833 on a shared 500-example training/evaluation set. The later driver
exit was nonzero and there was no saved checkpoint. This is historical
integration evidence, not a test of the new source or a long-run quality claim.

Quantized checkpoints, MTP, adapter-only/delta synchronization and resumability
are outside this full-parameter change. No SFT result is inferred from RL logs.
