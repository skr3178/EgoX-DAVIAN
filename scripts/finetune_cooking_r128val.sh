#!/bin/bash
# Cooking QLoRA — rank 128 (faster/less-overfit) + forward-only val loss on a fixed held-out set.
# Resolution stays LOW (49x176x704, the YAML default) for speed — no height increase.
# 24 GB: NF4 base + 8-bit AdamW. Val loss every 100 steps over VAL_META (deterministic, comparable).
set -eo pipefail
cd "$(dirname "$0")/.."          # -> EgoX/

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export EGOX_NF4=1                 # NF4 4-bit transformer
export EGOX_8BIT_OPTIM=1          # 8-bit AdamW (rank-128 still safer at 24 GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=localhost MASTER_PORT=29521

CT=/media/skr/SeagateHub1/egoexo4d/cooking_train
# Clean split (build_clean_split.py): 174 OK-prior clips -> 153 train / 21 held-out val,
# stratified by take (no scene leakage); black/dark priors excluded.
TRAIN_META="$CT/meta_train_clean.json"
VAL_META="$CT/meta_val_clean.json"

/media/skr/storage/conda_envs/egox/bin/accelerate launch --config_file configs_acc/1gpu.yaml --num_processes 1 --num_machines 1 \
  finetune.py \
    --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
    --model_name wan-i2v --model_type wan-i2v \
    --training_type lora --rank 128 --lora_alpha 128 \
    --output_dir ./results/EgoX_cooking_r128 \
    --report_to tensorboard \
    --data_root "$CT" \
    --meta_data_file "$TRAIN_META" \
    --validation_meta_file "$VAL_META" \
    --val_loss_steps 100 \
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
