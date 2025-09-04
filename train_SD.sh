#!/bin/bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="data/d7p_filtered"
export OUTPUT_DIR="baseline"


accelerate launch external/diffusers/examples/text_to_image/train_text_to_image.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$TRAIN_DIR \
--use_ema \
--resolution=256 --center_crop --random_flip \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--mixed_precision="fp16" \
--max_train_steps=15000 \
--learning_rate=1e-05 \
--max_grad_norm=1 \
--checkpointing_steps=7500 \
--lr_scheduler="constant" --lr_warmup_steps=0 \
--output_dir=${OUTPUT_DIR}
