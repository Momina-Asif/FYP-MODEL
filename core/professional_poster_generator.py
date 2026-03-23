"""
Professional Poster Generator - Product-Focused Layout
Generates posters with hero image + text overlay (like the examples)
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Tuple, Optional
import logging

from models.sd_base import load_sd
from utils.content_generator_v2 import ContentGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfessionalPosterGenerator:
    """
    Generates professional product posters with:
    - Hero product image (center)
    - Title at top
    - Description below image
    - CTA at bottom
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = 512
        
        logger.info("Loading Stable Diffusion...")
        self.sd_pipe = load_sd()
        
        self.content_gen = ContentGenerator()
        logger.info("✅ Poster generator initialized")

    def generate(
        self,
        cta: str,
        target_audience: str,
        product_description: str,
        style: str = "modern",
        output_path: Optional[str] = None
    ) -> Image.Image:
        """
        Generate a professional poster with hero image layout.
        
        Args:
            cta: Call-to-action text
            target_audience: Target audience description
            product_description: Product description (used for image generation)
            style: Visual style
            output_path: Path to save poster
            
        Returns:
            PIL Image of the poster
        """
        logger.info(f"Generating professional poster for: {product_description}")
        
        # Step 1: Generate product image
        logger.info("📷 Generating product image...")
        product_image = self._generate_product_image(product_description, style)
        
        # Step 2: Generate text content
        logger.info("✍️ Generating text content...")
        content_map = self.content_gen.generate_content(
            cta=cta,
            product_description=product_description,
            target_audience=target_audience,
            style=style
        )
        
        # Step 3: Create poster layout
        logger.info("🎨 Creating professional layout...")
        poster = self._create_professional_layout(
            product_image, content_map, product_description
        )
        
        # Step 4: Save if path provided
        if output_path:
            poster.save(output_path)
            logger.info(f"💾 Poster saved to {output_path}")
        
        logger.info("✅ Poster generation complete!")
        return poster

    def _generate_product_image(self, product_description: str, style: str) -> Image.Image:
        """Generate a product-focused image using Stable Diffusion"""
        prompt = f"professional product photo of {product_description}, {style} style, high quality, centered, white background, clean, well-lit, studio photography"
        
        logger.info(f"SD Prompt: {prompt}")
        
        with torch.no_grad():
            images = self.sd_pipe(
                prompt=prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images
        
        return images[0]

    def _create_professional_layout(
        self, 
        product_image: Image.Image,
        content_map: Dict[str, str],
        product_description: str
    ) -> Image.Image:
        """
        Create a professional poster layout:
        - Black background
        - Title at top (white)
        - Product image in center (prominent)
        - Description below image
        - CTA button at bottom
        """
        # Create canvas (taller for better text placement)
        width, height = 728, 1024
        poster = Image.new('RGB', (width, height), color=(20, 20, 20))
        draw = ImageDraw.Draw(poster)
        
        # Load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 64)
            subtitle_font = ImageFont.truetype("arial.ttf", 36)
            desc_font = ImageFont.truetype("arial.ttf", 26)
            cta_font = ImageFont.truetype("arial.ttf", 40)
            small_font = ImageFont.truetype("arial.ttf", 20)
        except:
            title_font = subtitle_font = desc_font = cta_font = small_font = ImageFont.load_default()
        
        # Extract content
        title = content_map.get("title", "Premium Product")[:50]
        subtitle = content_map.get("subtitle", "Experience Excellence")[:60]
        description = content_map.get("description", "Discover what makes us special")[:80]
        cta = content_map.get("cta", "Buy Now")
        
        y_pos = 0
        
        # ===== TITLE SECTION (Top) =====
        title_y = 40
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        
        # Draw title with effect
        for adj_x, adj_y in [(i, j) for i in [-2, -1, 0, 1, 2] for j in [-2, -1, 0, 1, 2]]:
            draw.text((title_x + adj_x, title_y + adj_y), title, fill=(0, 0, 0), font=title_font)
        draw.text((title_x, title_y), title, fill=(255, 255, 255), font=title_font)
        
        # ===== PRODUCT IMAGE SECTION (Center) =====
        # Resize product image to fit nicely
        product_img = product_image.copy()
        product_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
        
        # Center it
        img_x = (width - product_img.width) // 2
        img_y = 200  # Below title
        
        # Add subtle border/frame around product
        frame = Image.new('RGB', (product_img.width + 20, product_img.height + 20), color=(80, 80, 100))
        frame.paste(product_img, (10, 10))
        poster.paste(frame, (img_x - 10, img_y - 10))
        
        # ===== SUBTITLE SECTION (Below image) =====
        subtitle_y = img_y + product_img.height + 40
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = (width - subtitle_width) // 2
        
        draw.text((subtitle_x, subtitle_y), subtitle, fill=(255, 220, 100), font=subtitle_font)
        
        # ===== DESCRIPTION SECTION =====
        desc_y = subtitle_y + 60
        # Wrap description
        wrapped_desc = self._wrap_text(description, 40)
        
        desc_bbox = draw.textbbox((0, 0), wrapped_desc, font=desc_font)
        desc_width = desc_bbox[2] - desc_bbox[0]
        desc_x = (width - desc_width) // 2 if desc_width <= width - 40 else 40
        
        for line in wrapped_desc.split('\n'):
            line_bbox = draw.textbbox((0, 0), line, font=desc_font)
            line_width = line_bbox[2] - line_bbox[0]
            line_x = (width - line_width) // 2
            draw.text((line_x, desc_y), line, fill=(200, 200, 200), font=desc_font)
            desc_y += 40
        
        # ===== CTA BUTTON (Bottom) =====
        cta_y = height - 120
        
        # Draw button background
        button_padding = 30
        cta_bbox = draw.textbbox((0, 0), cta, font=cta_font)
        button_width = (cta_bbox[2] - cta_bbox[0]) + button_padding * 2
        button_height = (cta_bbox[3] - cta_bbox[1]) + button_padding
        
        button_x = (width - button_width) // 2
        button_y = cta_y
        
        # Button rectangle (bright color)
        draw.rectangle(
            [button_x, button_y, button_x + button_width, button_y + button_height],
            fill=(255, 100, 80),  # Coral/orange
            outline=(255, 150, 120)
        )
        
        # CTA text (white, centered in button)
        cta_text_x = button_x + button_padding + 15
        cta_text_y = button_y + (button_height - (cta_bbox[3] - cta_bbox[1])) // 2
        draw.text((cta_text_x, cta_text_y), cta, fill=(255, 255, 255), font=cta_font)
        
        return poster

    def _wrap_text(self, text: str, max_chars: int = 40) -> str:
        """Wrap text to fit width"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = " ".join(current_line + [word])
            if len(test_line) <= max_chars:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n".join(lines)


def generate_professional_poster(
    cta: str,
    target_audience: str,
    product_description: str,
    style: str = "modern",
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Simple wrapper to generate a professional poster.
    
    Usage:
        poster = generate_professional_poster(
            cta="Buy Now",
            target_audience="Coffee lovers",
            product_description="Premium espresso machine",
            style="luxury",
            output_path="output/poster.png"
        )
    """
    generator = ProfessionalPosterGenerator()
    return generator.generate(
        cta=cta,
        target_audience=target_audience,
        product_description=product_description,
        style=style,
        output_path=output_path
    )
