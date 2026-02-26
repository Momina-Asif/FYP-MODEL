import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image, ImageDraw
from models.layout_autoencoder import LayoutAutoEncoder

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
dataset = load_dataset("creative-graphic-design/CGL-Dataset")
small_val = dataset["validation"].select(range(20))  # small subset for testing

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Dataset class
class CGLPosterDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        # Always return tensor
        img_tensor = self.transform(item["image"].convert("RGB")) if self.transform else transforms.ToTensor()(item["image"])

        anns = item["annotations"]
        if isinstance(anns, str):
            import json
            anns = json.loads(anns)
        if isinstance(anns, dict):
            anns = [anns]
        if isinstance(anns, list) and len(anns) > 0 and isinstance(anns[0], str):
            import json
            anns = [json.loads(a) for a in anns]

        boxes = []
        _, height, width = img_tensor.shape
        for ann in anns:
            if "bbox" not in ann:
                continue
            bbox = ann["bbox"]
            if isinstance(bbox[0], list):
                bbox = bbox[0]
            x, y, w, h = bbox
            boxes.append([x / width, y / height, w / width, h / height])

        if len(boxes) == 0:
            boxes = [[0, 0, 0, 0]]

        return img_tensor, torch.tensor(boxes, dtype=torch.float32)

# DataLoader
val_ds = CGLPosterDataset(small_val, transform=transform)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)  # batch size=1

# Load model
model = LayoutAutoEncoder().to(device)
model.load_state_dict(torch.load("layout_autoencoder.pth", map_location=device))
model.eval()
print("Model loaded.")

# Testing loop
# Testing loop
for idx, (img_tensor, boxes) in enumerate(val_loader):
    boxes = boxes.to(device)  # pass boxes, not image
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        pred_boxes = model(boxes)  # ✅ model expects boxes
        pred_boxes = pred_boxes.squeeze(0).cpu()  # remove batch dimension

    # Convert image tensor to PIL for visualization
    img_pil = transforms.ToPILImage()(img_tensor.squeeze(0).cpu())
    draw = ImageDraw.Draw(img_pil)
    _, height, width = img_tensor.shape[1:]

    # Draw predicted boxes
    for box in pred_boxes:
        x, y, w, h = box
        x1, y1 = x * width, y * height
        x2, y2 = (x + w) * width, (y + h) * height
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    img_pil.show()

