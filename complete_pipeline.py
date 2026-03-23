#!/usr/bin/env python
"""
Complete Pipeline: Text (T5) → Image (Stable Diffusion) → Layout (Poster)
"""

import torch
import os
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from io import BytesIO
import numpy as np

from t5_inference import get_generator

# Load object detection model (YOLOv5)
model = None  # Initialize to None
try:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False)
    model.conf = 0.5  # Confidence threshold
    print("✅ YOLOv5 object detection model loaded")
except Exception as e:
    print(f"⚠️ Could not load YOLOv5: {e}, using default text positioning")
    model = None

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from PIL import Image, ImageDraw, ImageFont

import math
from PIL import Image, ImageDraw, ImageFont

def draw_cta_tilted_panel(poster, placement="top-right", panel_size=(500, 60),
                          panel_color=(255,0,0,255), cta_text="Click Here",
                          cta_text_color=(255,255,255,255),
                          font_path="arial.ttf", font_size=32,
                          corner_margin=0):

    if placement not in ["top-left", "top-right"]:
        return poster

    width, height = poster.size

    angle = 45 if placement == "top-left" else -45
    rad = math.radians(abs(angle))
    new_width = int(panel_size[0] * math.cos(rad) + panel_size[1] * math.sin(rad))
    new_panel_size = (new_width, panel_size[1])

    panel = Image.new("RGBA", new_panel_size, panel_color)

    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0,0), cta_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(
        ((new_panel_size[0]-text_width)//2, (new_panel_size[1]-text_height)//2),
        cta_text,
        fill=cta_text_color,
        font=font
    )

    rotated_panel = panel.rotate(angle, expand=True)
    r_w, r_h = rotated_panel.size

    # Position panel so it sticks out DRAMATICALLY from the corner
    if placement == "top-left":
        # Push far out of top-left corner (negative x and y)
        pos = (-r_w // 2+ 100, -r_h // 2 + 100)  # Extends halfway out
    else:  # top-right
        # Push far out of top-right corner (negative y, far right x)
        pos = (width - r_w // 2, -r_h // 2)  # Extends halfway out

    poster.paste(rotated_panel, pos, rotated_panel)
    return poster
# ============================================================================
# REGION-BASED DYNAMIC TEXT PLACEMENT SYSTEM
# ============================================================================

def create_regions(width, height):
    """
    Create a 3x3 grid of regions covering the entire image
    Returns list of 9 regions: [(x_min, y_min, x_max, y_max, region_name), ...]
    """
    col_width = width // 3
    row_height = height // 3
    
    regions = []
    names = [
        'top-left', 'top-center', 'top-right',
        'mid-left', 'mid-center', 'mid-right',
        'bot-left', 'bot-center', 'bot-right'
    ]
    
    for row in range(3):
        for col in range(3):
            x_min = col * col_width
            y_min = row * row_height
            x_max = (col + 1) * col_width if col < 2 else width
            y_max = (row + 1) * row_height if row < 2 else height
            region_name = names[row * 3 + col]
            regions.append((x_min, y_min, x_max, y_max, region_name))
    
    return regions


def calculate_overlap(product_bbox, region):
    """
    Calculate overlap area between product bounding box and region
    Returns overlap percentage (0.0 to 1.0)
    """
    p_x_min, p_y_min, p_x_max, p_y_max = product_bbox['x_min'], product_bbox['y_min'], product_bbox['x_max'], product_bbox['y_max']
    r_x_min, r_y_min, r_x_max, r_y_max = region[:4]
    
    # Calculate intersection
    inter_x_min = max(p_x_min, r_x_min)
    inter_y_min = max(p_y_min, r_y_min)
    inter_x_max = min(p_x_max, r_x_max)
    inter_y_max = min(p_y_max, r_y_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0  # No overlap
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    region_area = (r_x_max - r_x_min) * (r_y_max - r_y_min)
    
    return inter_area / region_area if region_area > 0 else 0.0


def get_best_region_for_text(product_bbox, regions, padding=30):
    """
    Find the region with least overlap with product
    Returns (best_region, available_width, available_height)
    """
    best_region = None
    min_overlap = 1.0
    
    for region in regions:
        overlap = calculate_overlap(product_bbox, region)
        if overlap < min_overlap:
            min_overlap = overlap
            best_region = region
    
    if best_region is None:
        best_region = regions[4]  # Fallback to center
    
    # Calculate available space for text
    r_x_min, r_y_min, r_x_max, r_y_max = best_region[:4]
    available_width = r_x_max - r_x_min - (2 * padding)
    available_height = r_y_max - r_y_min - (2 * padding)
    
    return best_region, available_width, available_height


def calculate_dynamic_font_size(available_width, text, base_font_size, font, min_size=20):
    """
    Dynamically adjust font size based on available width
    Returns adjusted font size
    """
    font_size = base_font_size
    
    while font_size > min_size:
        try:
            test_font = load_font(font, font_size)
            bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), text, font=test_font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= available_width:
                return font_size
        except:
            pass
        
        font_size -= 2
    
    return min_size


def place_text_in_region(draw, text, region, available_width, available_height, font_size, font, color, padding=30):
    """
    Place text centered in a region
    Returns the Y position after text for next element placement
    """
    r_x_min, r_y_min, r_x_max, r_y_max = region[:4]
    
    # Calculate text position (centered in region)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    text_x = r_x_min + padding + (available_width - text_width) // 2
    text_y = r_y_min + padding + (available_height - text_height) // 2
    
    # Draw text
    draw.text((text_x, text_y), text, fill=color, font=font)
    
    return text_x, text_y, text_height


# ============================================================================
# 0: OBJECT DETECTION (YOLOv5)
# ============================================================================

def detect_product_location(image_path):
    """
    Detect product/object location in image using YOLOv5
    Returns bounding box: (x_min, y_min, x_max, y_max, confidence, class)
    """
    try:
        # Check if model is loaded
        if model is None:
            print(f"      ⚠️ YOLOv5 model not available, using default positioning")
            return None
        
        print(f"      🔍 Detecting product location...")
        img = Image.open(image_path)
        
        # Run YOLOv5 detection
        results = model(img)
        detections = results.xyxy[0].cpu().numpy()  # Get detections
        
        if len(detections) == 0:
            print(f"      ⚠️ No objects detected, using default positioning")
            return None
        
        # Get the detection with highest confidence (main product)
        best_detection = detections[np.argsort(-detections[:, 4])[0]]  # Sort by confidence
        x_min, y_min, x_max, y_max, confidence, class_id = best_detection
        
        product_width = x_max - x_min
        product_height = y_max - y_min
        product_center_x = (x_min + x_max) / 2
        product_center_y = (y_min + y_max) / 2
        
        print(f"      ✅ Product detected:")
        print(f"         • Position: ({x_min:.0f}, {y_min:.0f}) to ({x_max:.0f}, {y_max:.0f})")
        print(f"         • Center: ({product_center_x:.0f}, {product_center_y:.0f})")
        print(f"         • Size: {product_width:.0f}x{product_height:.0f}")
        print(f"         • Confidence: {confidence:.2f}")
        
        return {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'confidence': confidence,
            'center_x': product_center_x,
            'center_y': product_center_y,
            'width': product_width,
            'height': product_height
        }
    
    except Exception as e:
        print(f"      ⚠️ Object detection failed: {e}, using default positioning")
        return None


def get_dynamic_text_positions(width, height, product_bbox):
    """
    Calculate optimal text positions based on product location and size
    Returns dict with positions for title, subtitle, description, button and layout info
    """
    if product_bbox is None:
        # Default positions - properly ordered: title -> subtitle -> description -> button
        return {
            'title_y': 20,
            'subtitle_y': 120,
            'description_y': 200,
            'button_y': height - 70,
            'title_x': 'center',
            'subtitle_x': 'center',
            'description_x': 'center',
            'strategy': 'default'
        }
    
    product_top = product_bbox['y_min']
    product_bottom = product_bbox['y_max']
    product_left = product_bbox['x_min']
    product_right = product_bbox['x_max']
    product_height = product_bbox['height']
    product_width = product_bbox['width']
    
    # Strategy 1: Product takes up most of vertical space (>60% height)
    if product_height > height * 0.6:
        print(f"      📍 TALL PRODUCT - Using RIGHT SIDE layout")
        # Position title and subtitle on TOP RIGHT
        # Position description and button on BOTTOM RIGHT
        return {
            'title_y': 20,
            'subtitle_y': 90,
            'description_y': height - 120,
            'button_y': height - 70,
            'title_x': 'right',
            'subtitle_x': 'right',
            'description_x': 'right',
            'right_margin': int(width * 0.15),  # Keep 15% margin from right
            'strategy': 'tall_product_right_side'
        }
    
    # Strategy 2: Product in upper half
    elif product_top < height * 0.4:
        print(f"      📍 Product in UPPER half - positioning text at BOTTOM")
        return {
            'title_y': int(product_bottom + 20),
            'subtitle_y': int(product_bottom + 100),
            'description_y': int(product_bottom + 160),
            'button_y': height - 70,
            'title_x': 'center',
            'subtitle_x': 'center',
            'description_x': 'center',
            'strategy': 'product_upper'
        }
    
    # Strategy 3: Product in lower half
    elif product_bottom > height * 0.6:
        print(f"      📍 Product in LOWER half - positioning text at TOP")
        return {
            'title_y': 20,
            'subtitle_y': 100,
            'description_y': 160,
            'button_y': max(int(product_top - 100), 20),
            'title_x': 'center',
            'subtitle_x': 'center',
            'description_x': 'center',
            'strategy': 'product_lower'
        }
    
    # Strategy 4: Product in middle but takes up significant width (>50% width)
    elif product_width > width * 0.5:
        print(f"      📍 WIDE PRODUCT - Using RIGHT SIDE layout")
        return {
            'title_y': 20,
            'subtitle_y': 90,
            'description_y': height - 120,
            'button_y': height - 70,
            'title_x': 'right',
            'subtitle_x': 'right',
            'description_x': 'right',
            'right_margin': int(width * 0.15),
            'strategy': 'wide_product_right_side'
        }
    
    # Default: center positioning
    else:
        print(f"      📍 Product in MIDDLE - positioning text centered")
        return {
            'title_y': 20,
            'subtitle_y': 100,
            'description_y': int(product_bottom + 30),
            'button_y': height - 70,
            'title_x': 'center',
            'subtitle_x': 'center',
            'description_x': 'center',
            'strategy': 'product_middle'
        }

# ============================================================================

def detect_product_category(text_output):
    """
    Detect product category from T5-generated text
    Returns category name for font selection
    """
    combined_text = f"{text_output.get('title', '')} {text_output.get('subtitle', '')} {text_output.get('description', '')}".lower()
    
    # Category keywords mapping
    categories = {
        'tech': ['headphones', 'earbuds', 'wireless', 'bluetooth', 'audio', 'speaker', 'phone', 'laptop', 'gadget', 'digital', 'smart', 'tech', 'electronic', 'device'],
        'fashion': ['style', 'fashion', 'bohemian', 'boho', 'chic', 'elegant', 'dress', 'pants', 'shirt', 'jacket', 'clothing', 'apparel', 'boutique', 'designer'],
        'luxury': ['premium', 'luxury', 'exclusive', 'haute', 'leather', 'designer', 'sophisticated', 'elegant', 'refined'],
        'sports': ['sport', 'athletic', 'gym', 'fitness', 'training', 'performance', 'energy', 'active'],
        'home': ['home', 'decor', 'furniture', 'cozy', 'comfort', 'living', 'interior', 'design'],
    }
    
    for category, keywords in categories.items():
        if any(keyword in combined_text for keyword in keywords):
            return category
    
    return 'default'


def get_fonts_for_category(category):
    """
    Select fonts based on product category
    Returns dict with title, subtitle, body, cta fonts
    """
    font_configs = {
        'tech': {
            'title': ('arial.ttf', 100, 'bold'),  # Clean, modern
            'subtitle': ('arial.ttf', 40, 'bold'),
            'body': ('arial.ttf', 26, 'regular'),
            'cta': ('arial.ttf', 36, 'bold'),
            'text_color': (255, 255, 255),  # White text for contrast
            'bg_overlay': (0, 0, 0, 180),  # Dark semi-transparent overlay
        },
        'fashion': {
            'title': ('georgia.ttf', 95, 'bold'),  # Elegant serif
            'subtitle': ('georgia.ttf', 38, 'italic'),
            'body': ('georgia.ttf', 24, 'regular'),
            'cta': ('georgia.ttf', 34, 'bold'),
            'text_color': (255, 255, 255),
            'bg_overlay': (0, 0, 0, 160),
        },
        'luxury': {
            'title': ('palatino.ttf', 100, 'bold'),  # Premium serif
            'subtitle': ('palatino.ttf', 40, 'regular'),
            'body': ('palatino.ttf', 25, 'regular'),
            'cta': ('palatino.ttf', 35, 'bold'),
            'text_color': (255, 215, 0),  # Gold text
            'bg_overlay': (0, 0, 0, 200),
        },
        'sports': {
            'title': ('impact.ttf', 110, 'bold'),  # Bold, energetic
            'subtitle': ('arial.ttf', 42, 'bold'),
            'body': ('arial.ttf', 27, 'bold'),
            'cta': ('impact.ttf', 38, 'bold'),
            'text_color': (255, 255, 255),
            'bg_overlay': (0, 0, 0, 170),
        },
        'home': {
            'title': ('georgia.ttf', 95, 'bold'),  # Warm, inviting
            'subtitle': ('georgia.ttf', 36, 'regular'),
            'body': ('georgia.ttf', 23, 'regular'),
            'cta': ('georgia.ttf', 32, 'bold'),
            'text_color': (255, 255, 255),
            'bg_overlay': (0, 0, 0, 150),
        },
        'default': {
            'title': ('arial.ttf', 100, 'bold'),
            'subtitle': ('arial.ttf', 40, 'bold'),
            'body': ('arial.ttf', 26, 'regular'),
            'cta': ('arial.ttf', 36, 'bold'),
            'text_color': (255, 255, 255),
            'bg_overlay': (0, 0, 0, 170),
        }
    }
    
    return font_configs.get(category, font_configs['default'])


def load_font(font_name, size):
    """
    Load font with fallback to Arial
    """
    try:
        return ImageFont.truetype(font_name, size)
    except:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            return ImageFont.load_default()


# ============================================================================
# 1: TEXT GENERATION (T5-BASE)
# ============================================================================

def generate_poster_text(product_context, use_base=True):
    """Generate marketing text using T5"""
    try:
        gen = get_generator("base" if use_base else "small")
        output = gen.generate(product_context, max_length=256)
        
        # Parse: Title | Subtitle | Description | CTA
        parts = output.split('|')
        
        if len(parts) >= 4:
            return {
                'title': parts[0].strip(),
                'subtitle': parts[1].strip(),
                'description': parts[2].strip(),
                'cta': parts[3].strip()
            }
        else:
            return {
                'title': output[:50].strip() if output else 'Premium Product',
                'subtitle': 'Premium Quality',
                'description': output,
                'cta': 'Shop Now'
            }
    except Exception as e:
        print(f"⚠️ Text generation failed: {e}")
        return {
            'title': 'Premium Product',
            'subtitle': 'High Quality',
            'description': f'Discover this remarkable {product_context}',
            'cta': 'Shop Now'
        }

# ============================================================================
# 2: IMAGE GENERATION (Stable Diffusion)
# ============================================================================

def generate_product_image(product_description, output_path, use_hf_api=True):
    """
    Generate image using HuggingFace Stable Diffusion API
    
    Args:
        product_description: Product to generate image for
        output_path: Where to save the image
        mode: 'product-focused' or 'balanced' (for prompt optimization)
        use_hf_api: Use HF API if True, else local GPU
    """
    
    try:
        print(f"   🎨 Generating image for: {product_description}")
        
        API_KEY = os.getenv("HF_API_KEY")
        if not API_KEY:
            print("   ⚠️ HF_API_KEY not set - creating placeholder")
            return create_placeholder_image(output_path)
        

        product = product_description.lower()

        # Default background
        background = "minimal luxury background with soft gradient and subtle abstract elements"

        # Category-based backgrounds
        if any(word in product for word in ["phone", "laptop", "headphones", "electronic", "camera"]):
            background = "futuristic tech environment with neon lights, reflections, and sleek surfaces"

        elif any(word in product for word in ["dress", "shirt", "fashion", "clothing", "jacket", "pants"]):
            background = "stylish fashion environment with soft studio lighting, aesthetic backdrop, and lifestyle vibe"

        elif "umbrella" in product:
            background = "rainy environment with falling raindrops, wet surfaces, and cinematic lighting"

        elif any(word in product for word in ["perfume", "fragrance"]):
            background = "luxury setting with marble textures, soft glow, and elegant shadows"

        elif any(word in product for word in ["shoes", "sneakers"]):
            background = "urban street style background with concrete textures and dynamic lighting"

        elif any(word in product for word in ["watch"]):
            background = "premium luxury setting with metallic textures and spotlight lighting"
        else:
            background = "minimal luxury background with spotlight lighting"
        # Final prompt
        prompt = f"""
        professional product photo of {product_description}, centered in frame,
        {background},
        dramatic cinematic lighting, high contrast, ultra-realistic reflections,
        sleek modern design, minimal luxury aesthetic,
        soft depth of field, glossy surfaces, studio-quality rendering,
        product hero shot composition with clear negative space at top and bottom,
        empty background space reserved for text overlay,
        balanced framing, rule of thirds, symmetrical layout,
        clean composition, no clutter,
        8k resolution, sharp focus, global illumination, ray tracing,
        photorealistic, cinematic color grading, high-end commercial ad style
        """
        model_url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-3-medium-diffusers"
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {"inputs": prompt}
        
        print(f"   🌐 Calling Stable Diffusion API...")
        response = requests.post(model_url, headers=headers, json=payload, timeout=180)
        
        if response.status_code != 200:
            print(f"   ⚠️ API error {response.status_code}: {response.text[:200]}")
            return create_placeholder_image(output_path)
        
        # Save image
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        image = Image.open(BytesIO(response.content))
        image.save(output_path)
        
        print(f"   ✅ Image generated: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"   ⚠️ Image generation failed: {e}")
        return create_placeholder_image(output_path)


def create_placeholder_image(output_path):
    """Create placeholder if generation fails"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    img = Image.new('RGB', (512, 512), color=(100, 150, 200))
    draw = ImageDraw.Draw(img)
    draw.text((150, 230), "PRODUCT IMAGE", fill='white')
    img.save(output_path)
    return output_path

# ============================================================================
# 3: ADAPTIVE COLOR DETECTION
# ============================================================================

def get_adaptive_colors(product_image_path):
    """
    Analyze image luminance and return adaptive text colors for optimal readability.
    
    Returns:
        dict: Color scheme with title_color, subtitle_color, body_color, button_color
    """
    try:
        # Load image and calculate average luminance
        img = Image.open(product_image_path).convert('RGB')
        img_small = img.resize((50, 50))  # Sample for speed
        pixels = list(img_small.getdata())
        
        # Calculate luminance for each pixel (weighted RGB)
        luminances = []
        for r, g, b in pixels:
            # Standard luminance formula
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            luminances.append(luminance)
        
        avg_luminance = sum(luminances) / len(luminances)
        
        print(f"      🌟 Image luminance: {avg_luminance:.2f}")
        
        # Define color schemes based on luminance
        if avg_luminance < 0.35:
            # DARK IMAGE - Use beige button and title
            print(f"      🌙 Detected: DARK IMAGE - Using beige button")
            return {
                'title_color': (176, 163, 134),      # Beige (matches button)
                'subtitle_color': (220, 220, 220),   # Light gray
                'body_color': (200, 200, 200),       # Light gray
                'button_color': (220, 210, 180),     # Beige (elegant on dark)
                'cta_text_color': (0, 0, 0)          # Black text on beige
            }
        elif avg_luminance > 0.65:
            # BRIGHT IMAGE - Use royal blue
            print(f"      ☀️ Detected: BRIGHT IMAGE - Using royal blue button")
            return {
                'title_color': (18, 37, 96),       # Royal blue (matches button)
                'subtitle_color': (60, 60, 80),      # Dark gray
                'body_color': (80, 80, 100),         # Medium dark
                'button_color': (18, 37, 96),      # Royal blue
                'cta_text_color': (255, 255, 255)    # White
            }
        else:
            # MEDIUM IMAGE - Use royal blue
            print(f"      🌤️ Detected: MEDIUM IMAGE - Using royal blue button")
            return {
                'title_color': (18, 37, 96),       # Royal blue (matches button)
                'subtitle_color': (100, 100, 100),   # Gray
                'body_color': (80, 80, 80),          # Dark gray
                'button_color': (18, 37, 96),      # Royal blue
                'cta_text_color': (255, 255, 255)    # White
            }
    
    except Exception as e:
        print(f"      ⚠️ Adaptive color detection failed: {e}, using defaults")
        # Fallback to balanced colors
        return {
            'title_color': (25, 118, 210),
            'subtitle_color': (100, 100, 100),
            'body_color': (80, 80, 80),
            'button_color': (200, 80, 180),
            'cta_text_color': (255, 255, 255)
        }

# ============================================================================
# 4: POSTER LAYOUT GENERATION
# ============================================================================

def create_professional_poster(product_text, product_image_path, output_path):
    """Create poster with text OVERLAID ON TOP of the product image"""
    
    try:
        print(f"   📄 Creating poster with text overlay ON image...")
        print(f"      🔍 DEBUG: product_text = {product_text}")
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Load product image
        product_img = Image.open(product_image_path).convert('RGB')
        width, height = product_img.width, product_img.height
        
        # Detect category from generated text
        category = detect_product_category(product_text)
        print(f"      📌 Detected category: {category}")
        
        # Get fonts based on category
        font_config = get_fonts_for_category(category)
        
        # Load category-specific fonts
        title_font = load_font(font_config['title'][0], font_config['title'][1])
        subtitle_font = load_font(font_config['subtitle'][0], font_config['subtitle'][1])
        body_font = load_font(font_config['body'][0], font_config['body'][1])
        cta_font = load_font(font_config['cta'][0], font_config['cta'][1])
        
        # GET ADAPTIVE COLORS BASED ON IMAGE
        adaptive_colors = get_adaptive_colors(product_image_path)
        title_color = adaptive_colors['title_color']
        subtitle_color = adaptive_colors['subtitle_color']
        body_color = adaptive_colors['body_color']
        button_color = adaptive_colors['button_color']
        cta_text_color = adaptive_colors['cta_text_color']
        
        # Use image directly without overlay
        final_poster = product_img.copy()
        draw = ImageDraw.Draw(final_poster)
        
        # Text positioning - place title at TOP with small margin
        margin = 25
        y_pos = 20  # Start at top with small margin
        center_x = width // 2
        
        # Prepare text content - USE EXACTLY WHAT T5 GENERATED
        title = product_text.get('title', 'Product')[:50].upper()
        subtitle = product_text.get('subtitle', 'Premium')[:50]
        description = product_text.get('description', 'Quality Product')[:100]  # USE T5 DESCRIPTION
        cta = product_text.get('cta', 'Shop Now').upper()
        
        print(f"      📝 POSTER TEXT:")
        print(f"         • Title: {title}")
        print(f"         • Subtitle: {subtitle}")
        print(f"         • Description: {description}")
        print(f"         • CTA: {cta}")
        
        # SMART TEXT PLACEMENT - Detect product and place text strategically
        print(f"      🎯 Using smart layout avoidance system")
        
        # Detect product location
        product_bbox = detect_product_location(product_image_path)
        print(f"      📍 Product bbox: {product_bbox}")
        
        # Convert final_poster to RGBA for compositing
        if final_poster.mode != 'RGBA':
            final_poster = final_poster.convert('RGBA')
        
        draw = ImageDraw.Draw(final_poster)
        
        # STRATEGY: PRIORITIZE TOP-MIDDLE REGION FIRST, then fallback to other regions
        # Priority: Top > Middle Regions > Side Regions > Center
        # Lowest Priority: Bottom regions
        
        if product_bbox:
            prod_center_x = product_bbox['center_x']
            prod_center_y = product_bbox['center_y']
            prod_width = product_bbox['width']
            prod_height = product_bbox['height']
            
            # Calculate space on each side
            space_left = prod_center_x - prod_width / 2
            space_right = width - (prod_center_x + prod_width / 2)
            space_top = prod_center_y - prod_height / 2
            space_bottom = height - (prod_center_y + prod_height / 2)
            
            print(f"      📏 Space - Left: {space_left:.0f}, Right: {space_right:.0f}, Top: {space_top:.0f}, Bottom: {space_bottom:.0f}")
            
            # Estimate title+subtitle height needed (rough estimate: 150-200px with margins)
            estimated_ts_height = 200
            
            # PRIORITY 1: TOP-CENTER region (preferred placement)
            if space_top > estimated_ts_height + 20:
                text_x_start = width // 2
                text_area_width = width * 0.85
                text_y = 20
                placement = "top"
                print(f"      ✅ [PRIORITY 1] Using TOP-CENTER placement (space_top: {space_top:.0f})")
            
            # PRIORITY 2: MIDDLE regions (left or right middle, not full side)
            elif space_right > 150 and space_top > 80:
                # Right side middle area
                text_x_start = prod_center_x + prod_width / 2 + 20
                text_area_width = space_right - 30
                text_y = max(30, int(prod_center_y - estimated_ts_height / 2))
                placement = "middle-right"
                print(f"      ✅ [PRIORITY 2] Using MIDDLE-RIGHT placement")
            
            elif space_left > 150 and space_top > 80:
                # Left side middle area
                text_x_start = 20
                text_area_width = space_left - 40
                text_y = max(30, int(prod_center_y - estimated_ts_height / 2))
                placement = "middle-left"
                print(f"      ✅ [PRIORITY 2] Using MIDDLE-LEFT placement")
            
            # PRIORITY 3: SIDE regions (full side height)
            elif space_right > 150:
                text_x_start = prod_center_x + prod_width / 2 + 20
                text_area_width = space_right - 30
                text_y = 30
                placement = "right"
                print(f"      ✅ [PRIORITY 3] Using RIGHT SIDE placement")
            
            elif space_left > 150:
                text_x_start = 20
                text_area_width = space_left - 40
                text_y = 30
                placement = "left"
                print(f"      ✅ [PRIORITY 3] Using LEFT SIDE placement")
            
            # PRIORITY 4: Fallback to center
            else:
                text_x_start = width * 0.1
                text_area_width = width * 0.8
                text_y = 30
                placement = "center"
                print(f"      ✅ [PRIORITY 4] Fallback to CENTER placement")
        else:
            text_x_start = width * 0.1
            text_area_width = width * 0.8
            text_y = 30
            placement = "center"
            print(f"      ℹ️ No product detected, using CENTER placement")
        
        print(f"      📍 Text placement: {placement}, X start: {text_x_start:.0f}, Width available: {text_area_width:.0f}")
        
        # Define padding early - needed for calculations
        text_padding = 30  # Pixels from edge
        
        # Calculate title size based on available width AND height
        # Especially important for TOP placement - constrain to available space above product
        available_height = height - text_padding
        
        if placement == "top" and product_bbox:
            # For TOP placement, constrain to space between image top and product top
            available_height = int(space_top - text_padding - 20)  # Leave 20px margin from product
            print(f"      📏 TOP placement - Available height: {available_height:.0f}px from top edge to product")
        
        # Start with width-based calculation - use FULL available width, not width-40
        # This allows titles on side placements to be larger
        calc_width = text_area_width if placement not in ["top", "center"] else text_area_width - 40
        title_size = calculate_dynamic_font_size(calc_width, title, 100, font_config['title'][0], min_size=48)
        subtitle_size = int(title_size * 0.4)  # Subtitle is 40% of title (60:40 ratio)
        
        # If TOP placement, also constrain based on available vertical space
        if placement == "top" and available_height > 0:
            # Estimate: title (1-2 lines) + subtitle (1-2 lines) + spacing
            # Line height roughly = font_size * 1.3
            estimated_line_height = title_size * 1.3 + 8 + (title_size * 0.4) * 1.3 + 8
            
            # If it doesn't fit, shrink the title
            while estimated_line_height > available_height and title_size > 48:
                title_size -= 4
                subtitle_size = int(title_size * 0.4)
                estimated_line_height = title_size * 1.3 + 8 + (title_size * 0.4) * 1.3 + 8
            
            print(f"      📏 Height-constrained - Title: {title_size}px, Subtitle: {subtitle_size}px (Est. height: {estimated_line_height:.0f}px)")
        
        title_font = load_font(font_config['title'][0], title_size)
        subtitle_font = load_font(font_config['subtitle'][0], max(subtitle_size, 20))
        
        print(f"      📝 Title size: {title_size}px, Subtitle size: {subtitle_size}px (60:40 ratio)")
        
        # WRAP TITLE TO 2 LINES IF NEEDED
        title_lines = []
        title_words = title.split()
        current_line = ""
        max_title_width = text_area_width - 20  # Reduced padding to allow larger fonts
        
        for word in title_words:
            test_line = f"{current_line} {word}".strip()
            test_bbox = draw.textbbox((0, 0), test_line, font=title_font)
            test_width = test_bbox[2] - test_bbox[0]
            
            if test_width > max_title_width and current_line:
                title_lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            title_lines.append(current_line)
        
        # WRAP SUBTITLE TO 2 LINES IF NEEDED
        subtitle_lines = []
        subtitle_words = subtitle.split()
        current_line = ""
        max_subtitle_width = text_area_width - 20  # Reduced padding to allow larger fonts
        
        for word in subtitle_words:
            test_line = f"{current_line} {word}".strip()
            test_bbox = draw.textbbox((0, 0), test_line, font=subtitle_font)
            test_width = test_bbox[2] - test_bbox[0]
            
            if test_width > max_subtitle_width and current_line:
                subtitle_lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            subtitle_lines.append(current_line)
        
        # Add padding margins to text placement
        text_padding = 30  # Pixels from edge
        text_padding_top = 20
        
        # Adjust text placement boundaries with padding
        if placement in ["left", "right", "middle-left", "middle-right"]:
            title_x_start_pos = text_x_start
            # Add padding from edges
            if placement in ["right", "middle-right"]:
                title_x_start_pos = max(title_x_start_pos, width - text_area_width - text_padding)
            elif placement in ["left", "middle-left"]:
                title_x_start_pos = min(title_x_start_pos, text_padding)
        else:
            title_x_start_pos = width // 2
        
        current_y = text_y + text_padding_top  # Add ttext_padding_topop padding
        
        # TRACK TITLE/SUBTITLE BOUNDING BOX FOR LATER USE
        title_subtitle_bbox = {
            'x_min': float('inf'),
            'y_min': current_y,
            'x_max': 0,
            'y_max': 0
        }
        
        # Draw title lines with shadow
        shadow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        
        title_positions = []  # Track all title positions
        for line in title_lines[:2]:  # Max 2 lines
            # Shadow
            bbox = draw.textbbox((0, 0), line, font=title_font)
            line_height = bbox[3] - bbox[1]
            line_width = bbox[2] - bbox[0]
            
            if placement in ["left", "right", "middle-left", "middle-right"]:
                line_x = title_x_start_pos + max(text_padding, (text_area_width - line_width) // 2 - text_padding)
            else:
                line_x = width // 2 - line_width // 2
            
            # Constrain to image bounds
            line_x = max(text_padding, min(line_x, width - line_width - text_padding))
            
            # Track bounds
            title_positions.append((line_x, current_y, line_x + line_width, current_y + line_height))
            title_subtitle_bbox['x_min'] = min(title_subtitle_bbox['x_min'], line_x)
            title_subtitle_bbox['x_max'] = max(title_subtitle_bbox['x_max'], line_x + line_width)
            title_subtitle_bbox['y_max'] = current_y + line_height
            
            for adj_x in range(-5, 6):
                if adj_x == 0:
                    continue
                shadow_draw.text((line_x + adj_x + 3, current_y + 4), line, fill=(0, 0, 0, 220), font=title_font)
            
            current_y += line_height + 8
        
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(10))
        final_poster.paste(shadow_layer, (0, 0), shadow_layer)
        
        draw = ImageDraw.Draw(final_poster)
        
        # Draw title lines with strokes and fill
        current_y = text_y + text_padding
        for line in title_lines[:2]:
            bbox = draw.textbbox((0, 0), line, font=title_font)
            line_height = bbox[3] - bbox[1]
            line_width = bbox[2] - bbox[0]
            
            if placement in ["left", "right", "middle-left", "middle-right"]:
                line_x = title_x_start_pos + max(text_padding, (text_area_width - line_width) // 2 - text_padding)
            else:
                line_x = width // 2 - line_width // 2
            
            # Constrain to image bounds
            line_x = max(text_padding, min(line_x, width - line_width - text_padding))
            
            # Draw strokes
            for adj_x in range(-5, 6):
                if adj_x == 0:
                    continue
                draw.text((line_x + adj_x, current_y), line, fill=title_color, font=title_font)
            
            # Draw solid
            draw.text((line_x, current_y), line, fill=title_color, font=title_font)
            current_y += line_height + 8
        
        # Draw subtitle lines with proper spacing from title
        current_y += 18
        for line in subtitle_lines[:2]:
            bbox = draw.textbbox((0, 0), line, font=subtitle_font)
            line_height = bbox[3] - bbox[1]
            line_width = bbox[2] - bbox[0]
            
            if placement in ["left", "right", "middle-left", "middle-right"]:
                line_x = title_x_start_pos + max(text_padding, (text_area_width - line_width) // 2 - text_padding)
            else:
                line_x = width // 2 - line_width // 2
            
            # Constrain to image bounds
            line_x = max(text_padding, min(line_x, width - line_width - text_padding))
            
            # Update title/subtitle bounds
            title_subtitle_bbox['x_min'] = min(title_subtitle_bbox['x_min'], line_x)
            title_subtitle_bbox['x_max'] = max(title_subtitle_bbox['x_max'], line_x + line_width)
            title_subtitle_bbox['y_max'] = current_y + line_height
            
            draw.text((line_x, current_y), line, fill=subtitle_color, font=subtitle_font)
            current_y += line_height + 8
        
        print(f"      📍 Title/Subtitle region: ({title_subtitle_bbox['x_min']:.0f}, {title_subtitle_bbox['y_min']:.0f}) to ({title_subtitle_bbox['x_max']:.0f}, {title_subtitle_bbox['y_max']:.0f})")
        
        # DYNAMIC DESCRIPTION PLACEMENT - Find best region avoiding product and title/subtitle
        desc_lines = []
        words = description.split()
        current_line = ""
        max_desc_width = width * 0.8
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            test_bbox = draw.textbbox((0, 0), test_line, font=body_font)
            test_width = test_bbox[2] - test_bbox[0]
            if test_width > max_desc_width:
                if current_line:
                    desc_lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            desc_lines.append(current_line)
        
        # Calculate description height needed
        desc_total_height = 0
        for line in desc_lines[:3]:
            bbox = draw.textbbox((0, 0), line, font=body_font)
            desc_total_height += bbox[3] - bbox[1] + 8
        
        # Define candidate regions for description
        if product_bbox:
            prod_y_min = product_bbox['y_min']
            prod_y_max = product_bbox['y_max']
            prod_x_min = product_bbox['x_min']
            prod_x_max = product_bbox['x_max']
            
            # Calculate available space
            space_above = prod_y_min
            space_below = height - prod_y_max
            space_left = prod_x_min
            space_right = width - prod_x_max
            
            # Define regions to try for description
            # PRIORITY: MIDDLE > LOWER > (never on product)
            regions = []
            
            # Helper function to check if a region overlaps with product
            def region_overlaps_product(region_y_start, region_y_end, region_x_start, region_x_end, prod_y_min, prod_y_max, prod_x_min, prod_x_max):
                # Check Y overlap
                y_overlap = not (region_y_end < prod_y_min or region_y_start > prod_y_max)
                # Check X overlap
                x_overlap = not (region_x_end < prod_x_min or region_x_start > prod_x_max)
                return y_overlap and x_overlap
            
            # PRIORITY 1 & 2: MIDDLE REGIONS (safest, avoids product)
            # MIDDLE-RIGHT region (right side, middle area - BETWEEN top and bottom of product)
            if space_right > 120:
                middle_right_y_start = prod_y_min + 40
                middle_right_y_end = prod_y_max - 40
                middle_right_x_start = prod_x_max + 10
                middle_right_x_end = width - text_padding
                
                # Only add if region doesn't overlap with product
                if not region_overlaps_product(middle_right_y_start, middle_right_y_end, middle_right_x_start, middle_right_x_end, prod_y_min, prod_y_max, prod_x_min, prod_x_max):
                    regions.append({
                        'name': 'middle-right',
                        'y_start': middle_right_y_start,
                        'y_end': middle_right_y_end,
                        'x_start': middle_right_x_start,
                        'x_end': middle_right_x_end,
                        'priority': 1
                    })
            
            # MIDDLE-LEFT region (left side, middle area - BETWEEN top and bottom of product)
            if space_left > 120:
                middle_left_y_start = prod_y_min + 40
                middle_left_y_end = prod_y_max - 40
                middle_left_x_start = text_padding
                middle_left_x_end = prod_x_min - 10
                
                # Only add if region doesn't overlap with product
                if not region_overlaps_product(middle_left_y_start, middle_left_y_end, middle_left_x_start, middle_left_x_end, prod_y_min, prod_y_max, prod_x_min, prod_x_max):
                    regions.append({
                        'name': 'middle-left',
                        'y_start': middle_left_y_start,
                        'y_end': middle_left_y_end,
                        'x_start': middle_left_x_start,
                        'x_end': middle_left_x_end,
                        'priority': 2
                    })
            
            # PRIORITY 3 & 4: LOWER REGIONS (secondary - below product only)
            # BOTTOM-CENTER region (center, below product)
            if space_below > desc_total_height + 20:
                bottom_center_y_start = prod_y_max + 15
                bottom_center_y_end = height - text_padding
                bottom_center_x_start = text_padding
                bottom_center_x_end = width - text_padding
                
                if not region_overlaps_product(bottom_center_y_start, bottom_center_y_end, bottom_center_x_start, bottom_center_x_end, prod_y_min, prod_y_max, prod_x_min, prod_x_max):
                    regions.append({
                        'name': 'bottom-center',
                        'y_start': bottom_center_y_start,
                        'y_end': bottom_center_y_end,
                        'x_start': bottom_center_x_start,
                        'x_end': bottom_center_x_end,
                        'priority': 3
                    })
            
            # BOTTOM-RIGHT region (right side, STRICTLY below product)
            if space_right > 120 and space_below > desc_total_height + 20:
                bottom_right_y_start = prod_y_max + 15
                bottom_right_y_end = height - text_padding
                bottom_right_x_start = prod_x_max + 10
                bottom_right_x_end = width - text_padding
                
                if not region_overlaps_product(bottom_right_y_start, bottom_right_y_end, bottom_right_x_start, bottom_right_x_end, prod_y_min, prod_y_max, prod_x_min, prod_x_max):
                    regions.append({
                        'name': 'bottom-right',
                        'y_start': bottom_right_y_start,
                        'y_end': bottom_right_y_end,
                        'x_start': bottom_right_x_start,
                        'x_end': bottom_right_x_end,
                        'priority': 4
                    })
            
            # BOTTOM-LEFT region (left side, STRICTLY below product)
            if space_left > 120 and space_below > desc_total_height + 20:
                bottom_left_y_start = prod_y_max + 15
                bottom_left_y_end = height - text_padding
                bottom_left_x_start = text_padding
                bottom_left_x_end = prod_x_min - 10
                
                if not region_overlaps_product(bottom_left_y_start, bottom_left_y_end, bottom_left_x_start, bottom_left_x_end, prod_y_min, prod_y_max, prod_x_min, prod_x_max):
                    regions.append({
                        'name': 'bottom-left',
                        'y_start': bottom_left_y_start,
                        'y_end': bottom_left_y_end,
                        'x_start': bottom_left_x_start,
                        'x_end': bottom_left_x_end,
                        'priority': 5
                    })
            
            # Sort by priority and choose best region
            regions.sort(key=lambda r: r['priority'])
            best_region = regions[0] if regions else None
            
            if best_region:
                desc_placement = best_region['name']
                desc_region_x_min = best_region['x_start']
                desc_region_x_max = best_region['x_end']
                desc_region_y_min = best_region['y_start']
                desc_region_y_max = best_region['y_end']
                max_desc_width = best_region['x_end'] - best_region['x_start'] - 20
                desc_x_center = (best_region['x_start'] + best_region['x_end']) // 2
                
                # For MIDDLE regions (middle-right, middle-left), place description vertically centered
                # For BOTTOM regions, place at top of region
                if 'middle' in desc_placement:
                    # Vertically center the description in the middle region
                    region_height = best_region['y_end'] - best_region['y_start']
                    region_center_y = best_region['y_start'] + region_height // 2
                    desc_y_start = region_center_y - (desc_total_height // 2)
                    # Ensure it stays within region bounds
                    desc_y_start = max(desc_y_start, best_region['y_start'] + 10)
                    desc_y_start = min(desc_y_start, best_region['y_end'] - desc_total_height - 10)
                else:
                    # For bottom regions, place at top of region
                    desc_y_start = best_region['y_start']
                
                print(f"      ✅ Description region selected: {desc_placement} (priority {best_region['priority']}, Y: {desc_y_start:.0f}-{desc_y_start + desc_total_height:.0f})")
            else:
                # Fallback only if no safe region found
                desc_placement = 'fallback-bottom'
                desc_y_start = height * 0.7
                desc_region_x_min = text_padding
                desc_region_x_max = width - text_padding
                desc_region_y_min = desc_y_start
                desc_region_y_max = height - text_padding
                desc_x_center = width // 2
                max_desc_width = width - 60
                print(f"      ⚠️ No safe region found, using fallback at Y={desc_y_start:.0f}")
        else:
            desc_placement = 'center'
            desc_y_start = height * 0.65
            desc_region_x_min = text_padding
            desc_region_x_max = width - text_padding
            desc_region_y_min = desc_y_start
            desc_region_y_max = height - text_padding
            desc_x_center = width // 2
            max_desc_width = width - 60
        
        # Rewrap description with correct width
        desc_lines_final = []
        words = description.split()
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            test_bbox = draw.textbbox((0, 0), test_line, font=body_font)
            test_width = test_bbox[2] - test_bbox[0]
            if test_width > max_desc_width:
                if current_line:
                    desc_lines_final.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            desc_lines_final.append(current_line)
        
        # Draw description lines
        desc_y = desc_y_start
        desc_bbox_bounds = {'x_min': float('inf'), 'y_min': desc_y_start, 'x_max': 0, 'y_max': 0}
        
        for i, line in enumerate(desc_lines_final[:3]):
            desc_bbox = draw.textbbox((0, 0), line, font=body_font)
            desc_width = desc_bbox[2] - desc_bbox[0]
            desc_height = desc_bbox[3] - desc_bbox[1]
            
            # Center description within the selected region bounds
            desc_x = desc_x_center - desc_width // 2
            
            # STRICT enforcement: keep description within selected region
            # Lower bound: at least 10px from region start
            desc_x = max(desc_x, desc_region_x_min + 10)
            # Upper bound: ensure it doesn't exceed region end
            desc_x = min(desc_x, desc_region_x_max - desc_width - 10)
            # Also respect image edges
            desc_x = max(text_padding, desc_x)
            desc_x = min(desc_x, width - desc_width - text_padding)
            
            # Y position: keep within region bounds
            desc_y = min(desc_y, desc_region_y_max - desc_height)
            desc_y = max(desc_y, desc_region_y_min)
            
            desc_bbox_bounds['x_min'] = min(desc_bbox_bounds['x_min'], desc_x)
            desc_bbox_bounds['x_max'] = max(desc_bbox_bounds['x_max'], desc_x + desc_width)
            desc_bbox_bounds['y_max'] = desc_y + desc_height
            
            draw.text((desc_x, desc_y), line, fill=body_color, font=body_font)
            desc_y += desc_height + 8
        
        print(f"      📋 Description placed {desc_placement}, bounds: ({desc_bbox_bounds['x_min']:.0f}, {desc_bbox_bounds['y_min']:.0f}) to ({desc_bbox_bounds['x_max']:.0f}, {desc_bbox_bounds['y_max']:.0f})")
        
        # DYNAMIC BUTTON PLACEMENT - place in safest corner
        button_width = 160
        button_height = 50
        padding = 15
        
        # Try to place button in a corner that avoids product
        corners = [
            ("top-left", (padding, padding)),
            ("top-right", (width - button_width - padding, padding)),
            ("bottom-left", (padding, height - button_height - padding)),
            ("bottom-right", (width - button_width - padding, height - button_height - padding)),
        ]
        
        button_x, button_y = width // 2 - button_width // 2, height - button_height - 15
        best_corner = None
        
        def check_overlap(rect1, rect2):
            """Check if two rectangles overlap and return overlap ratio"""
            overlap_x = max(0, min(rect1['x_max'], rect2['x_max']) - max(rect1['x_min'], rect2['x_min']))
            overlap_y = max(0, min(rect1['y_max'], rect2['y_max']) - max(rect1['y_min'], rect2['y_min']))
            overlap_area = overlap_x * overlap_y
            rect_area = (rect1['x_max'] - rect1['x_min']) * (rect1['y_max'] - rect1['y_min'])
            return overlap_area / rect_area if rect_area > 0 else 0
        
        # Find corner with least overlap with product AND title/subtitle
        min_total_overlap = 2.0
        for corner_name, (cx, cy) in corners:
            button_rect = {'x_min': cx, 'y_min': cy, 'x_max': cx + button_width, 'y_max': cy + button_height}
            
            # Check product overlap
            prod_overlap = check_overlap(button_rect, product_bbox) if product_bbox else 0
            
            # Check title/subtitle overlap
            ts_overlap = check_overlap(button_rect, title_subtitle_bbox)
            
            # Check description overlap
            desc_overlap = check_overlap(button_rect, desc_bbox_bounds)
            
            total_overlap = prod_overlap + ts_overlap + desc_overlap
            
            if total_overlap < min_total_overlap:
                min_total_overlap = total_overlap
                best_corner = (cx, cy, corner_name)
        
        if best_corner:
            button_x, button_y, corner_name = best_corner
            print(f"      🔘 Button placed in {corner_name}: ({button_x:.0f}, {button_y:.0f}), overlap: {min_total_overlap*100:.0f}%")
        else:
            print(f"      🔘 Button using default position: ({button_x:.0f}, {button_y:.0f})")
        final_poster = draw_cta_tilted_panel(
            final_poster,
            placement= corner_name,   # You can choose top-left, top-right, bottom-left, bottom-right
            panel_size=(500, 80),
            panel_color=button_color,
            cta_text=cta,
            cta_text_color=cta_text_color,
            font_path=font_config['cta'][0],
            font_size=32
        )     

        # # Draw STADIUM-shaped button (rounded rectangle)
        # radius = button_height // 2  # Full height for perfect stadium
        
        # # Draw filled stadium shape with rounded corners
        # draw.ellipse(
        #     [(button_x, button_y), (button_x + radius * 2, button_y + button_height)],
        #     fill=button_color
        # )
        # draw.ellipse(
        #     [(button_x + button_width - radius * 2, button_y), (button_x + button_width, button_y + button_height)],
        #     fill=button_color
        # )
        # draw.rectangle(
        #     [(button_x + radius, button_y), (button_x + button_width - radius, button_y + button_height)],
        #     fill=button_color
        # )
        
        # # Use smaller CTA font for stadium button
        # small_cta_font = load_font(font_config['cta'][0], 18)
        # small_cta = cta[:15].upper()  # Shorten text if needed
        
        # # Draw CTA text (centered in stadium button)
        # cta_bbox = draw.textbbox((0, 0), small_cta, font=small_cta_font)
        # cta_width = cta_bbox[2] - cta_bbox[0]
        # cta_height = cta_bbox[3] - cta_bbox[1]
        # cta_x = button_x + (button_width - cta_width) // 2
        # cta_y = button_y + (button_height - cta_height) // 2
        # # Draw main text
        # draw.text((cta_x, cta_y), small_cta, fill=cta_text_color, font=small_cta_font)
        
        # Convert back to RGB if needed
        if final_poster.mode == 'RGBA':
            final_poster = final_poster.convert('RGB')
        
        final_poster.save(output_path, quality=95)
        print(f"   ✅ Poster created: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"   ⚠️ Poster creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# COMPLETE PIPELINE - MAIN ENTRY POINT
# ============================================================================

def generate_complete_poster(product_description, use_base=True):
    """
    🎯 MAIN FUNCTION - Runs complete pipeline
    
    Input: product_description (string), use_base (bool)
    
    Returns: {
        'text': {...},
        'image': 'path',
        'poster': 'path'
    }
    """
    
    print(f"\n{'='*80}")
    print(f"GENERATING POSTER: {product_description}")
    print(f"{'='*80}\n")
    
    os.makedirs('output', exist_ok=True)
    
    # Step 1: Generate text with T5
    print("📝 [1/3] Generating text with T5-BASE..." if use_base else "📝 [1/3] Generating text with T5-SMALL...")
    text_output = generate_poster_text(product_description, use_base=use_base)
    print(f"   ✅ {text_output['title']} | {text_output['cta']}")
    
    # Step 2: Generate image with Stable Diffusion
    image_path = f"output/image_{product_description.replace(' ', '_')[:30]}.png"
    print("\n🎨 [2/3] Generating image with Stable Diffusion...")
    actual_image_path = generate_product_image(product_description, image_path)
    
    # Step 3: Create poster layout
    poster_path = f"output/poster_{product_description.replace(' ', '_')[:30]}.png"
    print("\n📄 [3/3] Creating poster layout...")
    final_poster = create_professional_poster(text_output, actual_image_path, poster_path)
    
    print(f"\n{'='*80}")
    print("✅ POSTER GENERATED!")
    print(f"{'='*80}\n")
    
    return {
        'text': text_output,
        'image': actual_image_path,
        'poster': final_poster
    }

