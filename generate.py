from diffusers import StableDiffusionPipeline
import torch

model_path = '/home/jfayyad/Python_Projects/LesionGen/Lora/weights/checkpoint-15000'

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")
pipe.safety_checker = None

prompt = "melanoma small size on a dark skin tone"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("generated_images/example.png")
