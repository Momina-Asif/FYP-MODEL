import torch

IMG_SIZE = 256
VOCAB_SIZE = 1000
EMBED_DIM = 128

# Auto-detect device - use CUDA if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 512

NUM_BOXES = 10

NUM_CLASSES = 5
# 0 = title
# 1 = subtitle
# 2 = image
# 3 = description
# 4 = logo

