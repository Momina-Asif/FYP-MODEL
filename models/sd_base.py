import torch
from diffusers import StableDiffusionPipeline
from config import DEVICE

def load_sd():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None
    )
    pipe = pipe.to(DEVICE)
    
    # Only enable xformers if on GPU
    if DEVICE == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass  # xformers not installed or not available
    
    return pipe

if __name__ == "__main__":
    pipe = load_sd()
    image = pipe("modern sale poster, bold typography, clean layout").images[0]
    image.save("test_poster.png")
    print("Saved test_poster.png")
