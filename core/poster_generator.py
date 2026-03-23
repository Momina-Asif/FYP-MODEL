"""
Main Poster Generation Framework
Orchestrates the entire pipeline: Layout -> Background -> Text Rendering
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Dict, Tuple, Optional
import logging

from models.layout_model import LayoutModel
from models.text_to_layout import TextToLayout
from models.sd_base import load_sd
from utils.content_generator_v2 import ContentGenerator
from config import NUM_BOXES, NUM_CLASSES, DEVICE, IMAGE_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PosterGenerator:
    """
    Complete poster generation pipeline.
    Generates layout -> background -> text overlay -> final poster
    """

    def __init__(self, model_paths: Optional[Dict] = None):
        """
        Initialize the poster generator with all required models.
        
        Args:
            model_paths: Dictionary with paths to model weights
                Default paths used if not provided
        """
        self.device = DEVICE
        self.image_size = IMAGE_SIZE
        
        # Load models
        self._load_models(model_paths or {})
        
        # Initialize content generator
        self.content_gen = ContentGenerator()
        
        logger.info("✅ PosterGenerator initialized successfully")

    def _load_models(self, model_paths: Dict):
        """Load all required models"""
        logger.info("Loading models...")
        
        # Layout Model
        layout_path = model_paths.get("layout", "layout_model.pth")
        self.layout_model = LayoutModel().to(self.device)
        try:
            self.layout_model.load_state_dict(
                torch.load(layout_path, map_location=self.device)
            )
            logger.info(f"✅ Layout model loaded from {layout_path}")
        except FileNotFoundError:
            logger.warning(f"⚠️ Layout model not found at {layout_path}, using random initialization")
        self.layout_model.eval()

        # Stable Diffusion Pipeline
        logger.info("Loading Stable Diffusion... (this may take a moment)")
        self.sd_pipe = load_sd()
        logger.info("✅ Stable Diffusion loaded")

    def generate(
        self,
        cta: str,
        target_audience: str,
        product_description: str,
        style: str = "modern",
        output_path: Optional[str] = None
    ) -> Tuple[Image.Image, Dict]:
        """
        Generate a complete poster from structured content.
        
        Args:
            cta: Call-to-action text (e.g., "Join Now", "Buy Today")
            target_audience: Description of target audience
            product_description: Description of the product/event
            style: Visual style (modern, minimalist, vibrant, etc.)
            output_path: Save the poster to this path if provided
            
        Returns:
            Tuple of (PIL Image, metadata dictionary)
        """
        logger.info(f"Generating poster for: {product_description}")
        
        # Step 1: Generate layout
        logger.info("📐 Step 1: Generating layout...")
        layout_boxes, layout_classes = self._generate_layout(
            cta, target_audience, product_description
        )
        
        # Step 2: Generate background using Stable Diffusion
        logger.info("🎨 Step 2: Generating background...")
        background_prompt = self._create_sd_prompt(
            product_description, target_audience, style
        )
        background = self._generate_background(background_prompt)
        
        # Step 3: Render layout elements on background
        logger.info("✍️ Step 3: Rendering layout elements...")
        poster, metadata = self._render_poster(
            background, layout_boxes, layout_classes, cta, product_description,
            target_audience, style
        )
        
        # Step 4: Save if path provided
        if output_path:
            poster.save(output_path)
            logger.info(f"💾 Poster saved to {output_path}")
        
        logger.info("✅ Poster generation complete!")
        return poster, metadata

    def _generate_layout(
        self, cta: str, target_audience: str, product_description: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate layout boxes and classes using a structured approach.
        Creates a clean top-to-bottom layout: Title -> Subtitle -> Description -> CTA
        """
        # Create a structured layout instead of random boxes
        # Image divided into sections for different elements
        
        layout_boxes = []
        layout_classes = []
        
        # Title section (top 20% of poster) - class 0
        layout_boxes.append([0.1, 0.05, 0.8, 0.15])  # x, y, width, height
        layout_classes.append(0)  # title
        
        # Subtitle section (next 15%) - class 1
        layout_boxes.append([0.1, 0.22, 0.8, 0.12])
        layout_classes.append(1)  # subtitle
        
        # Description section (middle 25%) - class 3
        layout_boxes.append([0.1, 0.36, 0.8, 0.20])
        layout_classes.append(3)  # description
        
        # CTA section (prominent bottom 15%) - class 0 (treat as important)
        layout_boxes.append([0.15, 0.65, 0.7, 0.12])
        layout_classes.append(0)  # CTA with title-level prominence
        
        # Add padding with empty elements to reach NUM_BOXES
        for _ in range(len(layout_boxes), NUM_BOXES):
            layout_boxes.append([0.0, 0.0, 0.0, 0.0])  # Empty box
            layout_classes.append(4)  # logo class (won't be rendered if empty)
        
        boxes = np.array(layout_boxes, dtype=np.float32)
        classes = np.array(layout_classes, dtype=np.int32)
        
        logger.info(f"Generated structured layout with {len(boxes)} elements (4 active)")
        return boxes, classes

    def _create_sd_prompt(
        self, product_description: str, target_audience: str, style: str
    ) -> str:
        """
        Create an optimized prompt for Stable Diffusion background generation.
        """
        prompt_parts = [
            f"{style} poster background",
            f"for {target_audience}",
            f"featuring {product_description}",
            "professional design, high quality, modern, clean layout",
            "no text, suitable for overlay text"
        ]
        return ", ".join(prompt_parts)

    def _generate_background(self, prompt: str) -> Image.Image:
        """
        Generate background image using Stable Diffusion.
        """
        logger.info(f"SD Prompt: {prompt}")
        
        with torch.no_grad():
            images = self.sd_pipe(
                prompt=prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                height=self.image_size,
                width=self.image_size
            ).images
        
        background = images[0]
        logger.info("✅ Background generated")
        return background

    def _render_poster(
        self,
        background: Image.Image,
        boxes: np.ndarray,
        classes: np.ndarray,
        cta: str,
        product_description: str,
        target_audience: str = "",
        style: str = "modern"
    ) -> Tuple[Image.Image, Dict]:
        """
        Render text and layout elements on the background.
        Only renders the 4 main elements: title, subtitle, description, and CTA.
        """
        poster = background.copy().convert('RGB')
        
        # Try to load a nice font, fallback to default
        try:
            title_font = ImageFont.truetype("arial.ttf", 56)
            subtitle_font = ImageFont.truetype("arial.ttf", 32)
            text_font = ImageFont.truetype("arial.ttf", 24)
            cta_font = ImageFont.truetype("arial.ttf", 44)
        except:
            logger.warning("⚠️ Arial font not found, using default font")
            title_font = subtitle_font = text_font = cta_font = ImageFont.load_default()
        
        metadata = {
            "elements": [],
            "num_boxes": len(boxes)
        }
        
        # Generate content
        content_map = self.content_gen.generate_content(
            cta=cta,
            product_description=product_description,
            target_audience=target_audience,
            style=style
        )
        logger.info(f"Content map: {content_map}")
        
        # Only render the first 4 boxes (title, subtitle, description, CTA)
        render_count = 0
        for idx, (box, class_id) in enumerate(zip(boxes, classes)):
            # Skip empty boxes
            if box[2] == 0 or box[3] == 0:
                continue
            
            # Only render first 4 elements
            if render_count >= 4:
                break
            
            x, y, w, h = box
            x_px = int(x * self.image_size)
            y_px = int(y * self.image_size)
            w_px = int(w * self.image_size)
            h_px = int(h * self.image_size)
            
            element_type = self._get_class_name(class_id)
            
            # For CTA (4th element), always use the CTA text
            if render_count == 3:
                text_content = cta
                element_type = "cta"
            else:
                text_content = content_map.get(element_type, "")
            
            if text_content:
                logger.info(f"Rendering {element_type}: '{text_content}' at ({x_px}, {y_px})")
                
                # Create a drawing layer with dark semi-transparent background
                overlay = Image.new('RGBA', poster.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                # Draw dark semi-transparent rectangle for better text contrast
                overlay_draw.rectangle(
                    [x_px - 5, y_px - 5, x_px + w_px + 5, y_px + h_px + 5],
                    fill=(0, 0, 0, 130)
                )
                
                # Apply overlay to poster
                poster = Image.alpha_composite(
                    poster.convert('RGBA'), overlay
                ).convert('RGB')
                
                # Select appropriate font and color based on element type
                if element_type == "title":
                    font = title_font
                    color = (255, 255, 255)  # White
                elif element_type == "subtitle":
                    font = subtitle_font
                    color = (255, 220, 100)  # Light yellow
                elif element_type == "cta":
                    font = cta_font
                    color = (255, 80, 80)  # Bright red/orange for CTA emphasis
                else:  # description
                    font = text_font
                    color = (220, 220, 220)  # Light gray
                
                # Wrap text for display
                wrapped_text = self._wrap_text(text_content, 35)
                
                # Draw text on poster
                draw = ImageDraw.Draw(poster)
                
                # Calculate text start position (centered in the box with padding)
                text_y = y_px + 10
                for line in wrapped_text.split('\n'):
                    # Draw text with outline for better visibility
                    for adj_x, adj_y in [(i, j) for i in [-2, -1, 0, 1, 2] for j in [-2, -1, 0, 1, 2]]:
                        draw.text(
                            (x_px + 15 + adj_x, text_y + adj_y),
                            line,
                            fill=(0, 0, 0, 180),  # Black outline
                            font=font
                        )
                    # Draw main text on top
                    draw.text(
                        (x_px + 15, text_y),
                        line,
                        fill=color,
                        font=font
                    )
                    text_y += getattr(font, 'size', 30) + 12
                
                metadata["elements"].append({
                    "type": element_type,
                    "text": text_content,
                    "box": [x, y, w, h],
                    "position_px": [x_px, y_px, w_px, h_px]
                })
                
                logger.info(f"✅ Rendered {element_type}")
                render_count += 1
            else:
                logger.warning(f"⚠️ No content for element type: {element_type}")
        
        logger.info(f"Total elements rendered: {render_count}")
        return poster, metadata

    def _wrap_text(self, text: str, max_chars: int = 40) -> str:
        """
        Wrap text to fit within a width.
        """
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

    def _get_class_name(self, class_id: int) -> str:
        """Map class ID to human-readable name"""
        class_map = {
            0: "title",
            1: "subtitle",
            2: "image",
            3: "description",
            4: "logo"
        }
        return class_map.get(class_id, "unknown")


def generate_poster_simple(
    cta: str,
    target_audience: str,
    product_description: str,
    style: str = "modern",
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Simple wrapper function to generate a poster in one line.
    
    Args:
        cta: Call-to-action text
        target_audience: Target audience description
        product_description: Product/event description
        style: Visual style
        output_path: Optional path to save the poster
        
    Returns:
        PIL Image of the generated poster
    """
    generator = PosterGenerator()
    poster, _ = generator.generate(
        cta=cta,
        target_audience=target_audience,
        product_description=product_description,
        style=style,
        output_path=output_path
    )
    return poster
