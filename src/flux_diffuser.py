# %%
import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline

# %%
flux_pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16
).to("mps")
flux_pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU VRAM

prompt = "A snowy moonlit night with fairies dancing around a Shinto temple placing protective enchantments around the area."
fx_image = flux_pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
).images[0]
fx_image

# %%
# Stable Diffusion
sd_pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16
).to("mps")
# sd_pipe.enable_model_cpu_offload()
sd_image = sd_pipe(prompt).images[0]
sd_image

# %%
# HiDream
hi_mod = "HiDream-ai/HiDream-I1-Full"
