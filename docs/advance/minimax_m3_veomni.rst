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


Expert LoRA companion
---------------------

The expert-adapter extension additionally needs the generic native integration
from `verl #8307 <https://github.com/verl-project/verl/pull/8307>`_ (inspected at
``8f42981f4c3f2d1e8723e78f6842e3d38198b563``), the native reference context from
`VeOmni #1337 <https://github.com/ByteDance-Seed/VeOmni/pull/1337>`_ (inspected at
``3780b042594d8597b34e4fbc5e683de3aea2fac6``), and the companion SwiGLU-OAI
expert-LoRA change. Publish a combined source commit and regenerate the lock
before installation; none of these unmerged heads is an interchangeable release.

Set ``actor_rollout_ref.model.lora_rank=64``, ``lora_alpha=64``,
``target_modules=all-linear`` and ``lora.merge=True``. For this model the default
selects text attention and dense/shared MLP linears plus independent
expert gate/up/down adapters; the frozen sparse indexer, vision and lm_head
are excluded. Indexer targets are rejected because its projections run under
no_grad without a published indexer training objective. Explicit dense
target lists are preserved. For explicit vision/projector targets, original
frozen dense biases are exported with their public names and unchanged values.
Zero dropout, bias='none', FSDP2, mixed precision
and a non-None parameter compute dtype are required. DDP/unsharded mixed-storage
execution is rejected. Full merged weights are built in temporary storage and
exported without mutating actor/optimizer weights. Reference inference uses the
native disable_adapter context. Adapter checkpoint recovery is not validated.

For the template, append the following overrides (the script's final overrides
take precedence)::

    actor_rollout_ref.model.lora_rank=64
    actor_rollout_ref.model.lora_alpha=64
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.model.lora.merge=True

Historical rank/alpha64 LoRA used the complete 60-layer/128-expert model,
training EP4/SP4 on four nodes and rollout TP32/EP1. Two updates on the same
16 training/evaluation examples yielded training rewards 90/128 and 94/128;
greedy same-set evaluation changed 10/16 to 15/16. DSpark was enabled there.
Step3 stalled and checkpoint saving was disabled. It is not evidence of
generalization or accelerator validation of the new implementation.
