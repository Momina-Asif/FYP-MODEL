import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

def load_controlnet():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-seg",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    pipe.enable_xformers_memory_efficient_attention()
    return pipe
