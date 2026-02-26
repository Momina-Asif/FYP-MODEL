import torch
import torch.nn as nn

NUM_BOXES = 10       # MATCH your dataset padding
LATENT_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 4
INPUT_DIM = NUM_BOXES * (4 + NUM_CLASSES)

class LayoutAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: boxes → latent
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM)
        )


        # Decoder: latent → boxes
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_BOXES * 4)   # 40
        )

    def forward(self, boxes):
        # Flatten input if needed
        if boxes.dim() == 3:  # (batch, NUM_BOXES, 4)
            boxes = boxes.view(boxes.size(0), -1)
        z = self.encoder(boxes)
        out = self.decoder(z)
        return out
