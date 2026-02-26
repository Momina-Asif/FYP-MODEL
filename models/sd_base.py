import torch
from diffusers import StableDiffusionPipeline

def load_sd():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

if __name__ == "__main__":
    pipe = load_sd()
    image = pipe("modern sale poster, bold typography, clean layout").images[0]
    image.save("test_poster.png")
    print("Saved test_poster.png")
