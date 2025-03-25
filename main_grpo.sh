#!/bin/bash

set -x

source ~/anaconda3/etc/profile.d/conda.sh
conda activate casual

ray stop

DATE=$(date +%Y%m%d)
export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_NAMES=("Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-3B-Instruct")
ROLLOUT_N=16

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$HOME/data/casual/train.parquet \
        data.val_files=$HOME/data/casual/test.parquet \
        data.train_batch_size=8 \
        data.val_batch_size=32 \
        data.max_prompt_length=1024 \
        data.max_response_length=2048 \
        actor_rollout_ref.model.path=$MODEL_NAME \
        actor_rollout_ref.actor.optim.lr=4e-7 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size=16 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.grad_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=$ROLLOUT_N \
        actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['wandb'] \
        trainer.project_name='GRPO_casual_clear' \
        trainer.experiment_name=$(basename $MODEL_NAME)_$ROLLOUT_N \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.default_hdfs_dir=null \
        trainer.save_freq=50 \
        trainer.test_freq=5 \
        trainer.total_epochs=3 $@ 2>&1 | tee grpo_$(basename $MODEL_NAME)_${ROLLOUT_N}_${DATE}.log
done
