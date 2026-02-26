import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from models.layout_model import LayoutModel
from config import NUM_BOXES, NUM_CLASSES

print("🚀 Starting poster generation...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    0: "text",
    1: "image",
    2: "logo",
    3: "decoration"
}

# -------------------------
# Load trained model
# -------------------------
model = LayoutModel().to(DEVICE)
model.load_state_dict(torch.load("layout_autoencoder.pth", map_location=DEVICE))
model.eval()
print("✅ Model loaded")

# -------------------------
# Use random real-like layout input
# -------------------------
# For demo, we can use uniform random boxes between 0 and 1
# Shape: (1, NUM_BOXES*4)
input_boxes = torch.rand(1, NUM_BOXES * 4).to(DEVICE)

with torch.no_grad():
    pred_boxes, pred_classes = model(input_boxes)

# Correct shape
pred_boxes = pred_boxes[0].cpu().clamp(0, 1).numpy()      # (NUM_BOXES, 4)
pred_classes = pred_classes[0].argmax(-1).cpu().numpy()   # (NUM_BOXES,)

print("Generated layout!")

# -------------------------
# Create blank poster
# -------------------------
POSTER_SIZE = 512
poster = Image.new("RGB", (POSTER_SIZE, POSTER_SIZE), (255, 255, 255))
draw = ImageDraw.Draw(poster)

try:
    font = ImageFont.truetype("arial.ttf", 18)
except:
    font = ImageFont.load_default()

# -------------------------
# Sample texts
# -------------------------
sample_texts = [
    "AI Poster Design",
    "Creative Conference 2026",
    "Join Us Today",
    "Future of Design",
    "Innovation & Creativity"
]

# -------------------------
# Draw elements
# -------------------------
for box, cls in zip(pred_boxes, pred_classes):
    x, y, w, h = box
    x *= POSTER_SIZE
    y *= POSTER_SIZE
    w *= POSTER_SIZE
    h *= POSTER_SIZE
    x2 = x + w
    y2 = y + h
    element = CLASS_NAMES[int(cls)]

    if element == "text":
        draw.rectangle([x, y, x2, y2], outline="black", width=2)
        text = np.random.choice(sample_texts)
        draw.text((x + 5, y + 5), text, fill="black", font=font)

    elif element == "image":
        draw.rectangle([x, y, x2, y2], fill=(200, 220, 255))
        draw.text((x + 5, y + 5), "IMAGE", fill="black", font=font)

    elif element == "logo":
        draw.ellipse([x, y, x2, y2], fill=(255, 200, 200))
        draw.text((x + 5, y + 5), "LOGO", fill="black", font=font)

    else:
        draw.rectangle([x, y, x2, y2], outline="gray")

# -------------------------
# Save & show poster
# -------------------------
poster.save("generated_poster.png")
poster.show()
print("✅ Poster saved as generated_poster.png")
