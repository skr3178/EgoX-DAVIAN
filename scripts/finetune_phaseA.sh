#!/bin/bash
# Phase A — single-GPU QLoRA smoke train on 1 cooking clip (validate the loop fits 24 GB).
set -eo pipefail
cd "$(dirname "$0")/.."          # -> EgoX/

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1                          # avoid ~/.local torch leak
export EGOX_NF4=1                                  # gated NF4 4-bit transformer load
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=localhost MASTER_PORT=29501

EX=/media/skr/storage/paper_reproduction/egoX/EgoX/example/egoexo4D

/media/skr/storage/conda_envs/egox/bin/accelerate launch --config_file configs_acc/1gpu.yaml --num_processes 1 --num_machines 1 \
  finetune.py \
    --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
    --model_name wan-i2v --model_type wan-i2v \
    --training_type lora --rank 64 --lora_alpha 64 \
    --output_dir ./results/EgoX_phaseA \
    --report_to tensorboard \
    --data_root "$EX" \
    --meta_data_file "$EX/meta_train_smoke_1clip.json" \
    --train_resolution 49x256x704 \
    --train_epochs 20 \
    --seed 42 \
    --batch_size 1 --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --num_workers 4 --pin_memory True \
    --checkpointing_steps 10 --checkpointing_limit 2 \
    --gen_fps 30 --cos_sim_scaling_factor 1.0
echo "END: $(date)"
