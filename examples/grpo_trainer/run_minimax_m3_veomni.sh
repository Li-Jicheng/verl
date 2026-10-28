#!/usr/bin/env bash
# Ascend integration template; source dependencies and capacity must be prepared.
set -euo pipefail
: "${MODEL_PATH:?Set the unquantized full-size MiniMax M3 checkpoint path}"
: "${TRAIN_FILES:?Set a prepared training parquet file}"
: "${VAL_FILES:?Set a prepared validation parquet file}"
: "${NNODES:?Set the prepared Ray cluster node count}"
: "${NPUS_PER_NODE:?Set NPUs per node}"
: "${TRAIN_EP:?Set training expert parallel size}"
: "${TRAIN_SP:?Set training sequence parallel size}"
: "${ROLLOUT_TP:?Set rollout tensor parallel size}"
: "${ROLLOUT_EP:?Set rollout expert parallel size}"

python -m verl.trainer.main_ppo --config-path=./config --config-name=ppo_trainer \
    model_engine=veomni algorithm.adv_estimator=grpo \
    "data.train_files=$TRAIN_FILES" "data.val_files=$VAL_FILES" \
    data.train_batch_size=16 data.max_prompt_length=1024 data.max_response_length=1024 \
    data.image_key=images data.filter_overlong_prompts=True data.truncation=error \
    "actor_rollout_ref.model.path=$MODEL_PATH" actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.veomni.mixed_precision=True \
    "actor_rollout_ref.actor.veomni.expert_parallel_size=$TRAIN_EP" \
    "actor_rollout_ref.actor.veomni.ulysses_parallel_size=$TRAIN_SP" \
    actor_rollout_ref.actor.veomni.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.veomni.moe_implementation=fused_npu \
    actor_rollout_ref.ref.strategy=veomni actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    "actor_rollout_ref.ref.veomni.expert_parallel_size=$TRAIN_EP" \
    "actor_rollout_ref.ref.veomni.ulysses_parallel_size=$TRAIN_SP" \
    actor_rollout_ref.ref.veomni.attn_implementation=flash_attention_2 \
    actor_rollout_ref.ref.veomni.moe_implementation=fused_npu \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=8 \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP" \
    "actor_rollout_ref.rollout.expert_parallel_size=$ROLLOUT_EP" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.enable_prefix_caching=False \
    "trainer.nnodes=$NNODES" "trainer.n_gpus_per_node=$NPUS_PER_NODE" \
    trainer.logger='[console]' trainer.project_name=minimax-m3-veomni \
    trainer.experiment_name=integration-template trainer.total_training_steps=2 \
    trainer.save_freq=-1 trainer.test_freq=1 trainer.resume_mode=disable "$@"
