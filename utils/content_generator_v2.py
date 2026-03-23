"""
Enhanced content generator using custom trained text generation model
Uses your own trainable Seq2Seq model for poster content
"""

import torch
from PIL import Image
from typing import Dict, Optional
import logging

from models.sd_base import load_sd
from utils.custom_text_inference import CustomTextGeneratorInference

logger = logging.getLogger(__name__)


class ContentGenerator:
    """
    Generates content (text and images) for poster elements.
    Uses custom trained text generation model.
    """

    def __init__(self, use_sd: bool = False, use_custom_text: bool = True):
        """
        Initialize content generator with custom text generation.
        
        Args:
            use_sd: Whether to use Stable Diffusion for element images
            use_custom_text: Whether to use custom trained text model
        """
        self.use_sd = use_sd
        self.use_custom_text = use_custom_text
        
        # Initialize custom text generator
        if use_custom_text:
            logger.info("Initializing custom text generator...")
            self.text_generator = CustomTextGeneratorInference()
        else:
            self.text_generator = None
        
        if use_sd:
            self.sd_pipe = load_sd()

    def generate_content(
        self, cta: str, product_description: str, target_audience: str = "", style: str = "modern"
    ) -> Dict[str, str]:
        """
        Generate text content for different poster elements using AI.
        
        Args:
            cta: Call-to-action text (e.g., "Join Now", "Buy Today")
            product_description: Description of product/event
            target_audience: Target audience description (optional)
            style: Visual style (modern, minimalist, vibrant, etc.)
            
        Returns:
            Dictionary mapping element types to AI-generated content
        """
        if self.text_generator:
            # Use AI-generated content
            content = {
                "title": self.text_generator.generate_title(product_description, style),
                "subtitle": self.text_generator.generate_subtitle(product_description),
                "description": self.text_generator.generate_description(product_description, target_audience),
                "cta": cta,
                "logo": ""
            }
            logger.info("✅ Generated content using AI")
        else:
            # Fallback to structured generation
            content = {
                "title": self._generate_title(product_description),
                "subtitle": self._generate_subtitle(product_description),
                "description": self._generate_description(product_description),
                "cta": cta,
                "logo": ""
            }
            logger.info("⚠️ Generated content using fallback methods (AI not available)")
        
        return content

    def _generate_title(self, product_description: str) -> str:
        """
        Fallback title - Extract compelling title words.
        E.g., "sleek water bottle" → "SLEEK HYDRATION"
        """
        words =  product_description.split()
        keywords_lower = product_description.lower()
        
        # Keyword-based title transformations
        title_map = {
            "water bottle": "STAY HYDRATED",
            "coffee": "COFFEE PERFECTED",
            "watch": "TIMELESS PRECISION",
            "phone": "CONNECTED LIFE",
            "camera": "CAPTURE MOMENTS",
            "headphone": "SOUND QUALITY",
            "shoe": "STEP FORWARD",
            "bag": "GO ANYWHERE",
        }
        
        for key, title in title_map.items():
            if key in keywords_lower:
                return title
        
        # Extract key adjectives + noun
        if "sleek" in keywords_lower:
            title = "SLEEK " + (words[-1].upper() if len(words) > 1 else "DESIGN")
        elif "premium" in keywords_lower:
            title = "PREMIUM " + (words[-1].upper() if len(words) > 1 else "QUALITY")
        elif len(words) >= 2:
            # First meaningful word + last word
            title = " ".join([w.upper() for w in words[:2] if len(w) > 2])
        else:
            title = product_description.upper()[:40]
        
        return title[:50]

    def _generate_subtitle(self, product_description: str) -> str:
        """
        Fallback subtitle - tagline based on product type.
        """
        desc_lower = product_description.lower()
        
        taglines = {
            "bottle": "Stay hydrated, stay focused",
            "water": "Pure hydration for active lifestyles",
            "coffee": "Your daily ritual perfected",
            "phone": "The latest technology at your fingertips",
            "watch": "Track every moment",
            "shoe": "Step into excellence",
            "bag": "Carry with confidence",
            "camera": "Capture every moment",
            "speaker": "Premium sound quality",
            "headphone": "Immersive audio experience",
            "laptop": "Power meets performance",
        }
        
        for keyword, tagline in taglines.items():
            if keyword in desc_lower:
                return tagline
        
        return "Experience the difference"

    def _generate_description(self, product_description: str) -> str:
        """
        Fallback description - Smart marketing copy based on keywords.
        Generates compelling ad copy relevant to the product.
        """
        keywords = product_description.lower()
        
        # Smart product-aware descriptions
        templates = {
            "water bottle": "Stay hydrated on every adventure",
            "bottle": "Premium quality you can trust",
            "sleek": "Modern design meets functionality",
            "coffee": "Authentic taste, perfectly crafted",
            "watch": "Precision engineered timepiece",
            "phone": "Latest technology in your hands",
            "camera": "Capture life in stunning detail",
            "headphone": "Immersive audio experience",
            "shoe": "Comfort and style combined",
            "bag": "Durable and stylish design",
            "smart": "Intelligent features for better living",
            "luxury": "Premium quality, exceptional value",
            "premium": "Crafted with excellence",
            "outdoor": "Built for adventure",
            "hiking": "Perfect for your next expedition",
            "travel": "Take it everywhere with ease",
            "portable": "Compact yet powerful",
            "lightweight": "Easy to carry, built tough",
        }
        
        # Check each keyword
        for key, desc in templates.items():
            if key in keywords:
                return desc
        
        # Default smart descriptions
        words = product_description.split()
        if len(words) > 1:
            return f"Experience the power of {words[0]} {words[1]}"
        
        return "Discover quality and performance"

    def generate_element_image(
        self, element_type: str, product_description: str, size: tuple = (256, 256)
    ) -> Image.Image:
        """
        Generate an image for a specific poster element.
        Uses Stable Diffusion if enabled.
        
        Args:
            element_type: Type of element (title, image, decoration, etc.)
            product_description: Product description for context
            size: Image size
            
        Returns:
            PIL Image
        """
        if not self.use_sd:
            # Return placeholder
            return Image.new("RGB", size, (200, 200, 200))
        
        prompt = self._create_element_prompt(element_type, product_description)
        
        logger.info(f"Generating {element_type} image: {prompt}")
        
        with torch.no_grad():
            images = self.sd_pipe(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=size[0],
                width=size[1]
            ).images
        
        return images[0]

    def _create_element_prompt(self, element_type: str, product_description: str) -> str:
        """
        Create optimized prompt for element image generation.
        """
        prompts = {
            "title": f"bold typography, {product_description}, professional design, high contrast",
            "image": f"high quality product image, {product_description}, professional photography",
            "decoration": f"decorative elements, geometric patterns, modern design, subtle, {product_description}",
            "logo": f"minimalist logo, {product_description}, professional, memorable",
        }
        
        return prompts.get(element_type, f"poster element for {product_description}")
