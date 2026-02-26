from PIL import Image, ImageDraw, ImageFont
from config import IMAGE_SIZE
from utils.content_generator import generate_text, generate_image

def render_poster(boxes, labels, prompt):

    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    draw = ImageDraw.Draw(canvas)

    font = ImageFont.load_default()

    for box, label in zip(boxes, labels):

        x, y, w, h = box
        x = int(x * IMAGE_SIZE)
        y = int(y * IMAGE_SIZE)
        w = int(w * IMAGE_SIZE)
        h = int(h * IMAGE_SIZE)

        if label == 2:  # IMAGE
            img = generate_image(prompt, (w, h))
            canvas.paste(img, (x, y))

        else:
            text = generate_text(label, prompt)
            draw.rectangle([x, y, x+w, y+h], outline="black")
            draw.text((x+5, y+5), text, fill="black", font=font)

    return canvas
