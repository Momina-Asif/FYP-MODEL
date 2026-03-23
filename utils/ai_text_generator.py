"""
AI-powered text generation for poster content.
Uses Hugging Face transformers for flexible, customizable text generation.
"""

import logging
from typing import Dict, Optional
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import torch

logger = logging.getLogger(__name__)


class AITextGenerator:
    """
    Generates poster text using AI models.
    Uses fine-tuned models for marketing/promotional content generation.
    """

    def __init__(self, model_name: str = "google/flan-t5-base", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the AI text generator.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run model on (cuda/cpu)
        """
        self.device = device
        self.model_name = model_name
        
        try:
            logger.info(f"Loading text generation model: {model_name}")
            # Use text2text generation pipeline (T5 based)
            self.generator = pipeline(
                "text2text-generation",
                model=model_name,
                device=0 if device == "cuda" else -1,
                max_length=512
            )
            logger.info("✅ AI Text Generator initialized successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.generator = None

    def generate_title(self, product_description: str, style: str = "modern", max_length: int = 50) -> str:
        """
        Generate a compelling title for the poster.
        
        Args:
            product_description: Description of product/event
            style: Visual style (modern, minimalist, bold, etc.)
            max_length: Maximum length of title
            
        Returns:
            Generated title text
        """
        if not self.generator:
            return self._fallback_title(product_description)
        
        prompt = f"""Generate a short, compelling, and catchy poster title for a {style} style poster.
Topic: {product_description}
Requirements:
- Maximum {max_length} characters
- All caps
- No punctuation
- Impactful and memorable
Title:"""
        
        try:
            result = self.generator(prompt, max_length=50, num_beams=3, early_stopping=True)
            title = result[0]['generated_text'].strip().upper()
            # Clean up and limit length
            title = title.replace("Title:", "").replace("Title", "").strip()[:max_length]
            logger.info(f"Generated title: '{title}'")
            return title
        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return self._fallback_title(product_description)

    def generate_subtitle(self, product_description: str, max_length: int = 60) -> str:
        """
        Generate a compelling subtitle that complements the title.
        
        Args:
            product_description: Description of product/event
            max_length: Maximum length of subtitle
            
        Returns:
            Generated subtitle text
        """
        if not self.generator:
            return self._fallback_subtitle(product_description)
        
        prompt = f"""Generate a concise, professional subtitle for a poster.
Main topic: {product_description}
Requirements:
- Maximum {max_length} characters
- Professional and informative
- Complements the main topic
Subtitle:"""
        
        try:
            result = self.generator(prompt, max_length=80, num_beams=3, early_stopping=True)
            subtitle = result[0]['generated_text'].strip()
            # Clean up output
            subtitle = subtitle.replace("Subtitle:", "").replace("Subtitle", "").strip()[:max_length]
            logger.info(f"Generated subtitle: '{subtitle}'")
            return subtitle
        except Exception as e:
            logger.error(f"Error generating subtitle: {e}")
            return self._fallback_subtitle(product_description)

    def generate_description(self, product_description: str, target_audience: str = "", max_length: int = 60) -> str:
        """
        Generate descriptive content for the poster.
        
        Args:
            product_description: Description of product/event
            target_audience: Target audience description
            max_length: Maximum length of description
            
        Returns:
            Generated description text
        """
        if not self.generator:
            return self._fallback_description(product_description)
        
        audience_str = f" for {target_audience}" if target_audience else ""
        prompt = f"""Generate a brief, engaging description for a poster{audience_str}.
Topic: {product_description}
Requirements:
- Maximum {max_length} characters
- Action-oriented and engaging
- Focus on benefits/value
- Professional tone
Description:"""
        
        try:
            result = self.generator(prompt, max_length=100, num_beams=3, early_stopping=True)
            description = result[0]['generated_text'].strip()
            # Clean up output
            description = description.replace("Description:", "").replace("Description", "").strip()[:max_length]
            logger.info(f"Generated description: '{description}'")
            return description
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return self._fallback_description(product_description)

    def generate_cta_text(self, cta_type: str = "generic", context: str = "") -> str:
        """
        Generate compelling call-to-action variations.
        
        Args:
            cta_type: Type of CTA (join, buy, register, discover, learn, etc.)
            context: Additional context for the CTA
            
        Returns:
            Generated CTA text
        """
        if not self.generator:
            return self._fallback_cta(cta_type)
        
        prompt = f"""Generate a short, punchy call-to-action text for a poster.
CTA type: {cta_type}
Context: {context if context else "General promotional"}
Requirements:
- 1-3 words maximum
- Action verb at the start
- Urgent/engaging tone
- Make it memorable
CTA:"""
        
        try:
            result = self.generator(prompt, max_length=30, num_beams=3, early_stopping=True)
            cta = result[0]['generated_text'].strip()
            # Clean up output
            cta = cta.replace("CTA:", "").replace("CTA", "").strip()
            logger.info(f"Generated CTA: '{cta}'")
            return cta
        except Exception as e:
            logger.error(f"Error generating CTA: {e}")
            return self._fallback_cta(cta_type)

    # Fallback methods for when AI generation fails
    def _fallback_title(self, product_description: str) -> str:
        """Fallback title generation"""
        words = product_description.split()
        if len(words) >= 2 and words[-1].lower() in ['summit', 'conference', 'expo', 'event', 'festival']:
            return " ".join(words[:-1] + [words[-1]]).upper()[:50]
        return " ".join(words[:2]).upper() if len(words) >= 2 else words[0].upper()

    def _fallback_subtitle(self, product_description: str) -> str:
        """Fallback subtitle generation"""
        if len(product_description) > 80:
            return product_description[:80] + "..."
        return product_description

    def _fallback_description(self, product_description: str) -> str:
        """Fallback description generation"""
        desc_phrases = {
            "limited": "Exclusive offer - Limited time only!",
            "sale": "Amazing deals on all items",
            "new": "Brand new collection available now",
            "event": "Join us for an amazing experience",
            "product": "Discover excellence and quality",
            "tech": "Cutting-edge technology innovation",
            "fashion": "Latest trends and styles",
        }
        
        for keyword, phrase in desc_phrases.items():
            if keyword in product_description.lower():
                return phrase
        
        return "Discover more and join us today!"

    def _fallback_cta(self, cta_type: str) -> str:
        """Fallback CTA generation"""
        cta_options = {
            "join": "Join Now",
            "buy": "Buy Today",
            "register": "Register Now",
            "discover": "Explore Now",
            "learn": "Learn More",
            "shop": "Shop Now",
            "subscribe": "Subscribe Now",
            "generic": "Act Now"
        }
        return cta_options.get(cta_type, "Act Now")
