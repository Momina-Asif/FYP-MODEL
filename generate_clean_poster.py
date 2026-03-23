"""Generate a CLEAN poster - 4 elements only, no overlap"""
from PIL import Image, ImageDraw, ImageFont
from utils.content_generator_v2 import ContentGenerator

# Create a nice gradient background
width, height = 512, 512
background = Image.new('RGB', (width, height))

# Create gradient (blue -> red)
for y in range(height):
    for x in range(width):
        r = int(50 + (150 * x / width))
        g = int(40)
        b = int(80 + (100 * (1 - y / height)))
        background.putpixel((x, y), (r, g, b))

# Generate content
content_gen = ContentGenerator()
content_map = content_gen.generate_content(
    cta="Join Now",
    product_description="Annual Tech Summit",
    target_audience="Tech Enthusiasts",
    style="modern"
)

# Get the 4 elements - take ONLY FIRST LINE
title = content_map.get("title", "TECH SUMMIT").split('\n')[0][:40]
subtitle = content_map.get("subtitle", "Experience Innovation").split('\n')[0][:40]
description = content_map.get("description", "Join industry leaders").split('\n')[0][:50]
cta = content_map.get("cta", "Join Now")

print(f"Title: {title}")
print(f"Subtitle: {subtitle}")
print(f"Description: {description}")
print(f"CTA: {cta}")

# Load fonts
try:
    title_font = ImageFont.truetype("arial.ttf", 56)
    subtitle_font = ImageFont.truetype("arial.ttf", 32)
    desc_font = ImageFont.truetype("arial.ttf", 20)
    cta_font = ImageFont.truetype("arial.ttf", 44)
except:
    title_font = subtitle_font = desc_font = cta_font = ImageFont.load_default()

draw = ImageDraw.Draw(background)

# Position 1: Title (top)
title_y = 40
# Draw shadow
draw.text((258, title_y + 3), title, fill=(0, 0, 0), font=title_font)
# Draw main text
draw.text((255, title_y), title, fill=(255, 255, 255), font=title_font)

# Position 2: Subtitle (below title)
subtitle_y = 140
draw.text((258, subtitle_y + 2), subtitle, fill=(0, 0, 0), font=subtitle_font)
draw.text((255, subtitle_y), subtitle, fill=(255, 220, 80), font=subtitle_font)

# Position 3: Description (middle)
desc_y = 240
draw.text((258, desc_y + 2), description, fill=(0, 0, 0), font=desc_font)
draw.text((255, desc_y), description, fill=(200, 200, 200), font=desc_font)

# Position 4: CTA (bottom)
cta_y = 390
draw.text((258, cta_y + 3), cta, fill=(0, 0, 0), font=cta_font)
draw.text((255, cta_y), cta, fill=(255, 100, 100), font=cta_font)

# Save
background.save('output/poster_clean.png')
print("\n✅ CLEAN poster saved to output/poster_clean.png")
print("\nLayout (4 elements only):")
print("  ├─ Title (white, top)")
print("  ├─ Subtitle (yellow, second)")
print("  ├─ Description (gray, middle)")
print("  └─ CTA (red, bottom)")
print("\nNO OVERLAP - CLEAN!")
