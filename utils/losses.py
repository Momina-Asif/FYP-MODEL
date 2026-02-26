import torch
import torch.nn.functional as F
from config import NUM_BOXES, NUM_CLASSES

def overlap_loss(boxes):
    loss = 0
    B = boxes.size(0)

    for b in range(B):
        for i in range(NUM_BOXES):
            for j in range(i+1, NUM_BOXES):

                xi1, yi1, wi1, hi1 = boxes[b, i]
                xi2, yi2, wi2, hi2 = boxes[b, j]

                xa = max(xi1, xi2)
                ya = max(yi1, yi2)
                xb = min(xi1+wi1, xi2+wi2)
                yb = min(yi1+hi1, yi2+hi2)

                inter = max(0, xb-xa) * max(0, yb-ya)
                loss += inter

    return loss / B


def classification_loss(pred_logits, labels):
    pred_logits = pred_logits.reshape(-1, NUM_CLASSES)
    labels = labels.view(-1)
    return F.cross_entropy(pred_logits, labels)
