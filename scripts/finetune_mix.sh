#!/bin/bash
# Mixed-domain QLoRA — cooking + diverse50 (16 domains, 552 clips), rank 128 + forward-only val loss.
# Cache + checkpoints on NVMe (/media/skr/storage/egoX_mix_train) for fast dataloading.
# Res 49x176x704 (YAML default; both datasets preencoded at this res). 24 GB: NF4 + 8-bit AdamW.
# ~2000 steps: 495 train / eff-batch 4 = ~124 steps/epoch x 16 epochs = ~1984 steps.
set -eo pipefail
cd "$(dirname "$0")/.."          # -> EgoX/

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export EGOX_NF4=1                 # NF4 4-bit transformer
export EGOX_8BIT_OPTIM=1          # 8-bit AdamW
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=localhost MASTER_PORT=29523

MIX=/media/skr/storage/egoX_mix_train          # NVMe unified dir (cache + metas + checkpoints)
TRAIN_META="$MIX/meta_train_mix.json"          # 495 clips, 16 domains
VAL_META="$MIX/meta_val_mix.json"              # 57 clips, stratified ~10% held-out

# Resume: RESUME=$MIX/results/EgoX_mix_r128/checkpoint-400 bash scripts/finetune_mix.sh
RESUME_ARG=""
if [ -n "${RESUME:-}" ]; then RESUME_ARG="--resume_from_checkpoint $RESUME"; echo "RESUMING from $RESUME"; fi

/media/skr/storage/conda_envs/egox/bin/accelerate launch --config_file configs_acc/1gpu.yaml --num_processes 1 --num_machines 1 \
  finetune.py \
    --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
    --model_name wan-i2v --model_type wan-i2v \
    --training_type lora --rank 128 --lora_alpha 128 \
    --output_dir "$MIX/results/EgoX_mix_r128" \
    --report_to tensorboard \
    --data_root "$MIX" \
    --meta_data_file "$TRAIN_META" \
    --validation_meta_file "$VAL_META" \
    --val_loss_steps 100 \
    --train_epochs 16 \
    --seed 42 \
    --batch_size 1 --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler constant_with_warmup --lr_warmup_steps 30 \
    --optimizer adamw --beta1 0.9 --beta2 0.95 --weight_decay 1e-4 --max_grad_norm 1.0 \
    --mixed_precision bf16 \
    --gradient_checkpointing True \
    --num_workers 8 --pin_memory True \
    --checkpointing_steps 200 --checkpointing_limit 12 \
    --gen_fps 30 --cos_sim_scaling_factor 1.0 $RESUME_ARG
echo "END: $(date)"
