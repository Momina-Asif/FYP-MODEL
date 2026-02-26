from PIL import Image
import random

def generate_text(label, prompt):

    if label == 0:
        return "TECH EXPO 2026"

    elif label == 1:
        return "Innovation Starts Here"

    elif label == 3:
        return "Join the biggest technology conference."

    return ""


def generate_image(prompt, size):

    # placeholder colored image
    img = Image.new(
        "RGB",
        size,
        (random.randint(50,200),
         random.randint(50,200),
         random.randint(50,200))
    )

    return img
