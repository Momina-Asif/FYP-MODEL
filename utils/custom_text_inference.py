"""
Inference wrapper for custom text generation model
Use this to generate different types of poster content
"""

import torch
import logging
import os
from typing import Optional, Dict
from models.custom_text_generator import PosterTextGenerator, Vocab

logger = logging.getLogger(__name__)


class CustomTextGeneratorInference:
    """
    Inference interface for the custom text generation model.
    Generates titles, subtitles, descriptions, and CTAs.
    """
    
    def __init__(self, model_path: str = "models/trained/poster_text_model.pth", 
                 vocab_path: str = "models/trained/vocab.pkl",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to trained model weights
            vocab_path: Path to vocabulary file
            device: Device to run model on
        """
        self.device = device
        self.vocab = Vocab()
        self.model = None
        self.model_loaded = False
        
        # Try to load the model
        self._load_model(model_path, vocab_path)
    
    def _load_model(self, model_path: str, vocab_path: str):
        """Load pre-trained model and vocabulary"""
        if not os.path.exists(model_path) or not os.path.exists(vocab_path):
            logger.warning(f"⚠️ Model not found at {model_path}")
            logger.warning("Using fallback text generation (no custom model)")
            self.model_loaded = False
            return
        
        try:
            # Load vocab
            self.vocab.load(vocab_path)
            logger.info(f"Loaded vocabulary with {len(self.vocab.word2idx)} words")
            
            # Load model
            vocab_size = len(self.vocab.word2idx)
            self.model = PosterTextGenerator(vocab_size=vocab_size).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            logger.info(f"✅ Loaded custom text generation model from {model_path}")
            self.model_loaded = True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False
    
    def _clean_text(self, text: str) -> str:
        """
        Clean generated text by removing metadata prefixes
        Remove patterns like "company:", "target audience:", etc.
        """
        # Remove common metadata prefixes
        prefixes_to_remove = [
            "company:", "target audience:", "constraints:", 
            "platform:", "campaign:", "objective:",
            "company ", "target ", "constraints "
        ]
        
        text = text.strip()
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove extra spaces and newlines
        text = " ".join(text.split())
        
        return text
    
    def generate_title(self, product_description: str, style: str = "modern", max_len: int = 15) -> str:
        """
        Generate a poster title.
        
        Args:
            product_description: Product/event description
            style: Visual style (context for generation)
            max_len: Maximum length of title in words
            
        Returns:
            Generated title
        """
        if not self.model_loaded:
            return self._fallback_title(product_description, style)
        
        try:
            # Prepare input context
            context = f"Generate title for {style} poster about {product_description}"
            input_ids = torch.tensor(
                self.vocab.encode(context, max_len=50),
                dtype=torch.long,
                device=self.device
            )
            
            # Generate
            title = self.model.generate(input_ids, max_len=max_len, vocab=self.vocab)
            title = title.replace("<END>", "").replace("<START>", "").strip()
            title = self._clean_text(title).upper()
            
            logger.info(f"Generated title: '{title}'")
            return title if title else self._fallback_title(product_description, style)
        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return self._fallback_title(product_description, style)
    
    def generate_subtitle(self, product_description: str, max_len: int = 20) -> str:
        """
        Generate a poster subtitle.
        
        Args:
            product_description: Product/event description
            max_len: Maximum length of subtitle in words
            
        Returns:
            Generated subtitle
        """
        if not self.model_loaded:
            return self._fallback_subtitle(product_description)
        
        try:
            context = f"Write subtitle for poster about {product_description}"
            input_ids = torch.tensor(
                self.vocab.encode(context, max_len=50),
                dtype=torch.long,
                device=self.device
            )
            
            subtitle = self.model.generate(input_ids, max_len=max_len, vocab=self.vocab)
            subtitle = subtitle.replace("<END>", "").replace("<START>", "").strip()
            subtitle = self._clean_text(subtitle)
            
            logger.info(f"Generated subtitle: '{subtitle}'")
            return subtitle if subtitle else self._fallback_subtitle(product_description)
        except Exception as e:
            logger.error(f"Error generating subtitle: {e}")
            return self._fallback_subtitle(product_description)
    
    def generate_description(self, product_description: str, target_audience: str = "", max_len: int = 20) -> str:
        """
        Generate a poster description.
        
        Args:
            product_description: Product/event description
            target_audience: Target audience description
            max_len: Maximum length of description
            
        Returns:
            Generated description
        """
        if not self.model_loaded:
            return self._fallback_description(product_description)
        
        try:
            audience_str = f" for {target_audience}" if target_audience else ""
            context = f"Write description{audience_str} about {product_description}"
            input_ids = torch.tensor(
                self.vocab.encode(context, max_len=50),
                dtype=torch.long,
                device=self.device
            )
            
            description = self.model.generate(input_ids, max_len=max_len, vocab=self.vocab)
            description = description.replace("<END>", "").replace("<START>", "").strip()
            description = self._clean_text(description)
            
            logger.info(f"Generated description: '{description}'")
            return description if description else self._fallback_description(product_description)
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return self._fallback_description(product_description)
    
    def generate_cta(self, context: str = "join", max_len: int = 5) -> str:
        """
        Generate a call-to-action.
        
        Args:
            context: Type of CTA (join, buy, register, etc.)
            max_len: Maximum length of CTA
            
        Returns:
            Generated CTA text
        """
        if not self.model_loaded:
            return self._fallback_cta(context)
        
        try:
            prompt = f"Generate short CTA for {context}"
            input_ids = torch.tensor(
                self.vocab.encode(prompt, max_len=50),
                dtype=torch.long,
                device=self.device
            )
            
            cta = self.model.generate(input_ids, max_len=max_len, vocab=self.vocab)
            cta = cta.replace("<END>", "").replace("<START>", "").strip()
            cta = self._clean_text(cta)
            
            logger.info(f"Generated CTA: '{cta}'")
            return cta if cta else self._fallback_cta(context)
        except Exception as e:
            logger.error(f"Error generating CTA: {e}")
            return self._fallback_cta(context)
    
    # Fallback methods when model is not loaded
    def _fallback_title(self, product_description: str, style: str = "modern") -> str:
        words = product_description.split()
        if len(words) >= 2 and words[-1].lower() in ['summit', 'conference', 'expo', 'event', 'festival']:
            return " ".join(words[:-1] + [words[-1]]).upper()[:50]
        return " ".join(words[:2]).upper() if len(words) >= 2 else words[0].upper()
    
    def _fallback_subtitle(self, product_description: str) -> str:
        return product_description[:80] + "..." if len(product_description) > 80 else product_description
    
    def _fallback_description(self, product_description: str) -> str:
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
    
    def _fallback_cta(self, context: str) -> str:
        cta_options = {
            "join": "Join Now",
            "buy": "Buy Today",
            "register": "Register Now",
            "discover": "Explore Now",
            "learn": "Learn More",
            "shop": "Shop Now",
            "subscribe": "Subscribe Now",
        }
        return cta_options.get(context, "Act Now")
