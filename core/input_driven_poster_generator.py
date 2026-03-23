"""
Professional Poster Generator v2 - Input-Driven
Takes structured input and generates cohesive professional posters
Layout: Image (left) | Text content (right)
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional
import logging

from models.sd_base import load_sd
from utils.content_generator_v2 import ContentGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputDrivenPosterGenerator:
    """
    Generates professional posters from structured input.
    Ensures product image matches description and text is coherent.
    
    Layout:
    - Image (left): Product photo, accurate and specific
    - Text (right): Title, subtitle, description, CTA
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Stable Diffusion...")
        self.sd_pipe = load_sd()
        self.content_gen = ContentGenerator()
        logger.info("✅ Generator initialized")

    def generate(
        self,
        cta: str,
        target_audience: str,
        product_description: str,
        style: str = "modern",
        output_path: Optional[str] = None
    ) -> Image.Image:
        """
        Generate a professional poster matching the input.
        
        Args:
            cta: Call-to-action (e.g., "Buy now")
            target_audience: Target audience (e.g., "outdoor hikers")
            product_description: Product description (e.g., "sleek water bottle")
            style: Visual style (modern, premium, minimalist, etc.)
            output_path: Path to save poster
            
        Returns:
            PIL Image of the poster
        """
        logger.info("=" * 60)
        logger.info("GENERATING PROFESSIONAL POSTER")
        logger.info(f"Product: {product_description}")
        logger.info(f"Target Audience: {target_audience}")
        logger.info(f"Style: {style}")
        logger.info("=" * 60)
        
        # Step 1: Generate ACCURATE product image
        logger.info("\n📷 STEP 1: Generating product image...")
        product_image = self._generate_product_image(product_description, style)
        
        # Step 2: Generate RELEVANT text
        logger.info("\n✍️ STEP 2: Generating text content...")
        content_map = self.content_gen.generate_content(
            cta=cta,
            product_description=product_description,
            target_audience=target_audience,
            style=style
        )
        
        # Step 3: Create professional side-by-side layout
        logger.info("\n🎨 STEP 3: Creating layout...")
        poster = self._create_side_by_side_layout(
            product_image,
            content_map,
            product_description,
            target_audience,
            cta,
            style
        )
        
        # Step 4: Save
        if output_path:
            poster.save(output_path)
            logger.info(f"✅ Poster saved to {output_path}")
        
        logger.info("=" * 60)
        logger.info("✅ POSTER GENERATION COMPLETE")
        logger.info("=" * 60)
        return poster

    def _generate_product_image(self, product_description: str, style: str) -> Image.Image:
        """
        Generate a SPECIFIC product image.
        
        Key: Use detailed, specific prompts to ensure the AI generates the RIGHT product
        """
        
        # Create detailed prompt that forces the model to generate the product
        detailed_prompt = f"""
        Professional product photography of a {product_description}.
        High quality, clean, {style} style.
        Single product, centered, on natural background.
        High resolution, well-lit, commercial quality.
        Focus on the product, no text, no people.
        """.strip()
        
        logger.info(f"SD Prompt: {detailed_prompt}")
        
        with torch.no_grad():
            images = self.sd_pipe(
                prompt=detailed_prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images
        
        product_img = images[0]
        logger.info("✅ Product image generated")
        return product_img

    def _create_side_by_side_layout(
        self,
        product_image: Image.Image,
        content_map: Dict[str, str],
        product_description: str,
        target_audience: str,
        cta: str,
        style: str
    ) -> Image.Image:
        """
        Create side-by-side layout:
        [Product Image] | [Text Content]
        
        Like the examples: image left, content right
        """
        
        # Canvas size (wider to accommodate side-by-side)
        canvas_width = 1024
        canvas_height = 768
        
        # Create dark background
        poster = Image.new('RGB', (canvas_width, canvas_height), color=(30, 30, 35))
        draw = ImageDraw.Draw(poster)
        
        # Load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 52)
            subtitle_font = ImageFont.truetype("arial.ttf", 28)
            desc_font = ImageFont.truetype("arial.ttf", 20)
            cta_font = ImageFont.truetype("arial.ttf", 36)
            label_font = ImageFont.truetype("arial.ttf", 16)
        except:
            title_font = subtitle_font = desc_font = cta_font = label_font = ImageFont.load_default()
        
        # ===== LEFT SIDE: PRODUCT IMAGE =====
        img_width = 450
        img_height = 650
        
        # Resize product image to fit
        product_img = product_image.copy()
        product_img.thumbnail((img_width - 40, img_height - 40), Image.Resampling.LANCZOS)
        
        # Add border/frame
        frame = Image.new('RGB', (img_width, img_height), color=(60, 60, 70))
        frame_x = (img_width - product_img.width) // 2
        frame_y = (img_height - product_img.height) // 2
        frame.paste(product_img, (frame_x, frame_y))
        
        # Paste frame to poster
        poster.paste(frame, (40, 60))
        
        # ===== RIGHT SIDE: TEXT CONTENT =====
        text_x = img_width + 100  # Start after image
        text_width = canvas_width - text_x - 40
        
        # Background box for text (olive/gold color like the example)
        text_bg_color = (120, 110, 60)  # Muted olive
        draw.rectangle(
            [text_x - 20, 50, canvas_width - 20, canvas_height - 50],
            fill=text_bg_color
        )
        
        # ADD BORDER/OUTLINE
        draw.rectangle(
            [text_x - 20, 50, canvas_width - 20, canvas_height - 50],
            outline=(150, 140, 90),
            width=2
        )
        
        y_offset = 80
        
        # Extract and clean content
        title = content_map.get("title", "Premium Product")[:50]
        subtitle = content_map.get("subtitle", "Excellence")[:50]
        description = content_map.get("description", "Experience quality")[:100]
        
        # ===== TITLE =====
        logger.info(f"Title: {title}")
        draw.text((text_x, y_offset), title, fill=(255, 255, 255), font=title_font)
        y_offset += 70
        
        # ===== DIVIDER =====
        draw.line([(text_x, y_offset), (canvas_width - 40, y_offset)], fill=(200, 190, 130), width=2)
        y_offset += 30
        
        # ===== SUBTITLE (Audience/Tagline) =====
        logger.info(f"Subtitle: {subtitle}")
        draw.text((text_x, y_offset), subtitle, fill=(255, 230, 150), font=subtitle_font)
        y_offset += 50
        
        # ===== PRODUCT HIGHLIGHT (what makes it special) =====
        highlight = f"Designed for {target_audience}"
        logger.info(f"Highlight: {highlight}")
        draw.text((text_x, y_offset), highlight, fill=(200, 200, 180), font=desc_font)
        y_offset += 45
        
        # ===== DESCRIPTION =====
        logger.info(f"Description: {description}")
        desc_lines = self._wrap_text(description, 45)
        for line in desc_lines.split('\n'):
            if line.strip():
                draw.text((text_x, y_offset), line, fill=(200, 200, 180), font=desc_font)
                y_offset += 35
        
        # ===== SPACER =====
        y_offset += 20
        
        # ===== CTA BUTTON =====
        button_y = canvas_height - 120
        
        # Button styling
        button_padding = 20
        cta_bbox = draw.textbbox((0, 0), cta, font=cta_font)
        button_width = (cta_bbox[2] - cta_bbox[0]) + button_padding * 2
        button_height = (cta_bbox[3] - cta_bbox[1]) + button_padding
        
        button_x = text_x + (text_width - button_width) // 2
        
        # Button background (coral/bright)
        draw.rectangle(
            [button_x, button_y, button_x + button_width, button_y + button_height],
            fill=(255, 120, 80),
            outline=(255, 150, 120),
            width=2
        )
        
        # CTA text
        cta_text_x = button_x + button_padding + 10
        cta_text_y = button_y + (button_height - (cta_bbox[3] - cta_bbox[1])) // 2
        logger.info(f"CTA: {cta}")
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


def generate_poster(
    cta: str,
    target_audience: str,
    product_description: str,
    style: str = "modern",
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Generate a professional poster from structured input.
    
    Usage:
        poster = generate_poster(
            cta="Buy now",
            target_audience="outdoor hikers",
            product_description="a sleek water bottle",
            style="modern",
            output_path="output/poster.png"
        )
    """
    generator = InputDrivenPosterGenerator()
    return generator.generate(
        cta=cta,
        target_audience=target_audience,
        product_description=product_description,
        style=style,
        output_path=output_path
    )
