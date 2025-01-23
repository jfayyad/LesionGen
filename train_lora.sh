#!/bin/bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="/home/jfayyad/Python_Projects/LesionGen/data/training_dataset"
export OUTPUT_DIR="Lora"

mkdir -p $OUTPUT_DIR

accelerate launch /home/jfayyad/Python_Projects/LesionGen/external/diffusers/examples/text_to_image/train_text_to_image_lora.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$TRAIN_DIR \
--resolution=256 --center_crop --random_flip \
--validation_prompt="generate a high quality image of melanoma skin lesion" \
--num_validation_images=1 \
--validation_epochs=5 \
--train_batch_size=1 \
--max_train_steps=15000 \
--learning_rate=1e-05 \
--lr_scheduler="constant" --lr_warmup_steps=0 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--mixed_precision="fp16" \
--rank=64 \
--prediction_type="epsilon" \
--output_dir=$OUTPUT_DIR
