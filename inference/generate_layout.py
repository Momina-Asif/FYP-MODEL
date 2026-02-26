import torch
from models.layout_model import LayoutModel
from config import *
from utils.renderer import render_poster

model = LayoutModel().to(DEVICE)
model.load_state_dict(torch.load("layout_model.pth"))
model.eval()

prompt = "Technology conference poster"

# dummy input
x = torch.randn(1, NUM_BOXES*4).to(DEVICE)

with torch.no_grad():
    pred_boxes, pred_classes = model(x)

boxes = pred_boxes.view(NUM_BOXES,4).cpu().numpy()
labels = pred_classes.view(NUM_BOXES,NUM_CLASSES).argmax(dim=1).cpu().numpy()

poster = render_poster(boxes, labels, prompt)
poster.save("final_poster.png")

print("✅ Poster Generated!")
