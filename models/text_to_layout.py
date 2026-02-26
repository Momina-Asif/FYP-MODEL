import torch
import torch.nn as nn
from models.text_encoder import TextEncoder

class TextToLayout(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_boxes):
        super().__init__()
        self.encoder = TextEncoder(vocab_size, embed_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8),
            num_layers=4
        )

        self.fc = nn.Linear(embed_dim, num_boxes * 5)

    def forward(self, tokens):
        x = self.encoder(tokens)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x).view(-1, 5)
