import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from models.layout_autoencoder import LayoutAutoEncoder
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
from config import *
from models.layout_model import LayoutModel
from utils.losses import overlap_loss, classification_loss

NUM_BOXES = 10
LABEL_MAP = {
    "text": 0,
    "image": 1,
    "logo": 2,
    "decoration": 3
}
NUM_CLASSES = len(LABEL_MAP)

def pad_boxes(boxes, max_boxes=NUM_BOXES):
    n = boxes.shape[0]
    if n >= max_boxes:
        return boxes[:max_boxes]
    pad = torch.zeros(max_boxes - n, 4)
    return torch.cat([boxes, pad], dim=0)


# -----------------------
# Dataset class
# -----------------------
class CGLPosterDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, max_boxes=10):
        self.ds = hf_dataset
        self.transform = transform
        self.max_boxes = max_boxes

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        if self.transform:
            img = self.transform(img)

        anns = item["annotations"]
        if isinstance(anns, str):
            anns = json.loads(anns)
        if isinstance(anns, dict):
            anns = [anns]
        if isinstance(anns, list) and len(anns) > 0 and isinstance(anns[0], str):
            anns = [json.loads(a) for a in anns]

        boxes = []
        labels = []

        _, height, width = img.shape

        for ann in anns:
            if "bbox" not in ann:
                continue
            bbox = ann["bbox"]
            if isinstance(bbox[0], list):
                    bbox = bbox[0]

            x, y, w, h = bbox

            # normalize box
            boxes.append([
                x / width,
                y / height,
                w / width,
                h / height
            ])
            category = ann.get("category", "decoration")

            # ---- FIX dataset inconsistencies ----
            if isinstance(category, list):
                category = category[0] if len(category) > 0 else "decoration"

            if not isinstance(category, str):
                category = "decoration"

            category = category.lower()

            if category not in LABEL_MAP:
                category = "decoration"


            labels.append(LABEL_MAP[category])

        if len(boxes) == 0:
            boxes = [[0, 0, 0, 0]]

        boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes = pad_boxes(boxes, self.max_boxes)
        labels = torch.tensor(labels, dtype=torch.long)

        # pad labels same as boxes
        if labels.shape[0] >= self.max_boxes:
            labels = labels[:self.max_boxes]
        else:
            pad = torch.zeros(self.max_boxes - labels.shape[0], dtype=torch.long)
            labels = torch.cat([labels, pad], dim=0)

        return img, boxes, labels

# -----------------------
# Load dataset
# -----------------------
dataset = load_dataset("creative-graphic-design/CGL-Dataset")
small_train = dataset["train"].select(range(500))
small_val = dataset["validation"].select(range(100))

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
])

train_ds = CGLPosterDataset(small_train, transform=transform)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

val_ds = CGLPosterDataset(small_val, transform=transform)
val_loader = DataLoader(val_ds, batch_size=8)

# -----------------------
# Model setup
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_BOXES = 10  # must match dataset
model = LayoutAutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.L1Loss()

# -----------------------
# Helper: sort boxes left → right
# -----------------------
def sort_boxes(boxes):
    """
    boxes: (B, NUM_BOXES, 4)
    sort using x coordinate
    """
    sorted_batches = []
    for b in range(boxes.size(0)):
        order = boxes[b, :, 0].argsort()
        sorted_batches.append(boxes[b][order])
    return torch.stack(sorted_batches)


# -----------------------
# Overlap penalty loss
# -----------------------
def overlap_loss(boxes):
    """
    Penalize overlapping boxes
    boxes shape: (B, NUM_BOXES, 4)
    """
    loss = 0.0
    B, N, _ = boxes.shape

    for b in range(B):
        for i in range(N):
            xi, yi, wi, hi = boxes[b, i]
            x1_i, y1_i = xi, yi
            x2_i, y2_i = xi + wi, yi + hi

            for j in range(i + 1, N):
                xj, yj, wj, hj = boxes[b, j]
                x1_j, y1_j = xj, yj
                x2_j, y2_j = xj + wj, yj + hj

                inter_w = torch.clamp(torch.min(x2_i, x2_j) - torch.max(x1_i, x1_j), min=0)
                inter_h = torch.clamp(torch.min(y2_i, y2_j) - torch.max(y1_i, y1_j), min=0)

                loss += inter_w * inter_h

    return loss / B


# -----------------------
# Training loop
# -----------------------

model = LayoutModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
bbox_loss_fn = nn.MSELoss()

for epoch in range(50):

    model.train()
    total_loss = 0

    for img, boxes, labels in train_loader:

        boxes = boxes.to(DEVICE)
        labels = labels.to(DEVICE)

        boxes_flat = boxes.view(boxes.size(0), -1)

        pred_boxes, pred_classes = model(boxes_flat)

        # bbox loss
        recon_loss = bbox_loss_fn(pred_boxes, boxes_flat)

        # overlap loss
        pred_boxes_reshaped = pred_boxes.view(-1, NUM_BOXES, 4)
        ov_loss = overlap_loss(pred_boxes_reshaped)

        # classification loss
        cls_loss = classification_loss(pred_classes, labels)

        loss = recon_loss + 0.5*ov_loss + 0.5*cls_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss {total_loss:.4f}")




torch.save(model.state_dict(), "layout_autoencoder.pth")
print("Model saved as layout_autoencoder.pth")

# -----------------------
# Evaluation & Visualization
# -----------------------
model.eval()
with torch.no_grad():
    for img, boxes, labels in val_loader:

        img = img.to(device)
        boxes = boxes.to(device)

        # same preprocessing as training
        boxes = sort_boxes(boxes)

        boxes_flat = boxes.view(boxes.size(0), -1).to(device)

        pred = model(boxes_flat)
        pred = pred.view(-1, NUM_BOXES, 4).cpu().clamp(0, 1)

        img_vis = img[0].permute(1, 2, 0).cpu().numpy()
        pred_boxes = pred[0].numpy()

        fig, ax = plt.subplots(1)
        ax.imshow(img_vis)

        for box in pred_boxes:
            x, y, w, h = box
            x *= img_vis.shape[1]
            y *= img_vis.shape[0]
            w *= img_vis.shape[1]
            h *= img_vis.shape[0]

            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)

        plt.show()
        break
