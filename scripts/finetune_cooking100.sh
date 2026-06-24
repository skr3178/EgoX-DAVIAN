#!/bin/bash
# 104-clip cooking QLoRA training — faithful EgoX defaults except the two forced 24 GB changes
# (res 49x256x704, NF4 base) + grad_accum 4 to recover the paper's effective batch of 4.
# Quick-signal: warmup 30. Stoppable/resumable (checkpoints every 100 steps).
set -eo pipefail
cd "$(dirname "$0")/.."          # -> EgoX/

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1                          # avoid ~/.local torch leak
export EGOX_NF4=1                                  # NF4 4-bit transformer
export EGOX_8BIT_OPTIM=1                           # 8-bit AdamW optimizer state (rank-256 fp32 AdamW OOMs at 24 GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=localhost MASTER_PORT=29520

CT=/media/skr/SeagateHub1/egoexo4d/cooking_train

/media/skr/storage/conda_envs/egox/bin/accelerate launch --config_file configs_acc/1gpu.yaml --num_processes 1 --num_machines 1 \
  finetune.py \
    --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
    --model_name wan-i2v --model_type wan-i2v \
    --training_type lora --rank 256 --lora_alpha 256 \
    --output_dir ./results/EgoX_cooking100 \
    --report_to tensorboard \
    --data_root "$CT" \
    --meta_data_file "$CT/meta_train_cooking.json" \
    --train_epochs 100 \
    --seed 42 \
    --batch_size 1 --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler constant_with_warmup --lr_warmup_steps 30 \
    --optimizer adamw --beta1 0.9 --beta2 0.95 --weight_decay 1e-4 --max_grad_norm 1.0 \
    --mixed_precision bf16 \
    --gradient_checkpointing True \
    --num_workers 8 --pin_memory True \
    --checkpointing_steps 100 --checkpointing_limit 10 \
    --gen_fps 30 --cos_sim_scaling_factor 1.0
echo "END: $(date)"
