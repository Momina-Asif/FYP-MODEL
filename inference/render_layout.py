import torch
from PIL import Image, ImageDraw

# generated boxes
boxes = torch.tensor([
    [0.4004, 0.4074, 0.4206, 0.4204],
    [0.5199, 0.4934, 0.3691, 0.3254],
    [0.4729, 0.5003, 0.3504, 0.3489],
])

# canvas
W, H = 512, 512
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

for box in boxes:
    x, y, w, h = box

    x1 = x * W
    y1 = y * H
    x2 = x1 + w * W
    y2 = y1 + h * H

    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

img.save("generated_layout.png")
img.show()
