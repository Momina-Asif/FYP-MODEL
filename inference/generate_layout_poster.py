from utils.layout_to_mask import poster_layout_mask
from models.controlnet_pipe import load_controlnet

pipe = load_controlnet()

layout = poster_layout_mask()
layout.save("layout.png")

prompt = "A modern electronics sale poster, bold title, clean design, high contrast"

image = pipe(prompt, image=layout).images[0]
image.save("layout_controlled_poster.png")

print("Saved layout_controlled_poster.png")
