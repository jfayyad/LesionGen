#!/bin/bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export TRAIN_DIR="../../../metadata/training_dataset"
export OUTPUT_DIR="Lora_SDXL"

accelerate launch train_text_to_image_lora_sdxl.py  \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --train_data_dir=$TRAIN_DIR \
  --train_batch_size=1 \
  --resolution=256 --center_crop --random_flip
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --lora_rank=8 \
  --use_8bit_adam \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=15000 \
  --checkpointing_steps=500 \
  --seed="0" 