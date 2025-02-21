import torch
import os
import random
import argparse
from diffusers import StableDiffusionXLPipeline

def generate_images(condition, mode, output_dir, num_images):
    model_path = "/home/jfayyad/Python_Projects/LesionGen/Lora/weights/checkpoint-301500"
    
    # Load the SDXL model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    )
    
    # Load LoRA weights
    pipe.load_lora_weights(model_path)
    
    # Move model to GPU
    pipe.to("cuda")
    
    # Disable safety checker (optional)
    pipe.safety_checker = None

    if mode == "single":
        single_output_dir = os.path.join(output_dir, "single")
        os.makedirs(single_output_dir, exist_ok=True)

        prompt = f"{condition} small size on a dark skin tone"
        image = pipe(prompt, num_inference_steps=100, guidance_scale=7.5, added_cond_kwargs={}).images[0]
        image.save(os.path.join(single_output_dir, f"{condition}.png"))

    elif mode == "dataset":
        dataset_output_dir = os.path.join(output_dir, "dataset", condition)
        os.makedirs(dataset_output_dir, exist_ok=True)

        prompts = [
            f"{condition} small size on a dark skin tone",
            f"{condition} large size on a light skin tone",
            f"{condition} medium size on a medium skin tone"
        ]
        guidance_scales = [3.0, 5.0, 7.5, 10.0]
        num_inference_steps_list = [30, 50, 70, 100]

        for i in range(num_images):
            prompt = random.choice(prompts)
            guidance_scale = random.choice(guidance_scales)
            num_inference_steps = random.choice(num_inference_steps_list)
            image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, added_cond_kwargs={}).images[0]
            image.save(os.path.join(dataset_output_dir, f"{i+1}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["single", "dataset"], required=True)
    parser.add_argument("--output_dir", type=str, default="generated_images")
    parser.add_argument("--num_images", type=int, default=10)
    args = parser.parse_args()

    generate_images(args.condition, args.mode, args.output_dir, args.num_images)
