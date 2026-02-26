import torch
import torch.nn as nn
from config import NUM_BOXES, NUM_CLASSES

class LayoutModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared MLP backbone
        self.net = nn.Sequential(
            nn.Linear(NUM_BOXES * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        # Output: boxes + class logits
        self.fc_out = nn.Linear(
            512,
            NUM_BOXES * 4 + NUM_BOXES * NUM_CLASSES
        )

    def forward(self, x):
        """
        x: tensor of shape (batch_size, NUM_BOXES*4)
        returns:
            box_out: (batch_size, NUM_BOXES, 4)
            class_out: (batch_size, NUM_BOXES, NUM_CLASSES)
        """
        x = self.net(x)
        out = self.fc_out(x)

        # split box coords and class logits
        box_out = out[:, :NUM_BOXES*4].reshape(-1, NUM_BOXES, 4)
        class_out = out[:, NUM_BOXES*4:].reshape(-1, NUM_BOXES, NUM_CLASSES)

        return box_out, class_out
