from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
import torch
from datetime import datetime
from PIL import Image

# Load models and pipeline
unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-ssd-1b", torch_dtype=torch.float16, variant="fp16")
pipe = DiffusionPipeline.from_pretrained("segmind/SSD-1B", unet=unet, torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
while True:
    prompt = input("Prompt: ")
    image_pil = pipe(prompt, num_inference_steps=5, guidance_scale=1.0).images[0]
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f"output-{current_time}.png"
    image_pil.save(save_path)