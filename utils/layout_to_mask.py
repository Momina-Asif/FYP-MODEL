import torch

def layout_to_mask(layout, img_size=256, num_classes=3):
    mask = torch.zeros((num_classes, img_size, img_size))

    for box in layout:
        x, y, w, h, cls = box
        cls = int(cls)
        x1 = int(x * img_size)
        y1 = int(y * img_size)
        x2 = int((x + w) * img_size)
        y2 = int((y + h) * img_size)
        mask[cls, y1:y2, x1:x2] = 1.0

    return mask
