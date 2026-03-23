#!/usr/bin/env python
"""
Advanced prompt engineering for Stable Diffusion product image generation.
Extracts product characteristics and generates optimized prompts for better results.
"""

import re

class ProductPromptBuilder:
    """Build optimized prompts for product image generation"""
    
    # Category detection
    ELECTRONICS = ['headphones', 'phone', 'laptop', 'tablet', 'watch', 'speaker', 'camera', 'keyboard', 'mouse']
    FASHION = ['shirt', 'dress', 'pants', 'shoes', 'jacket', 'coat', 'skirt', 'hat', 'backpack', 'wallet', 'purse']
    FURNITURE = ['chair', 'table', 'sofa', 'desk', 'bed', 'cabinet', 'shelf', 'lamp']
    HOME = ['cup', 'plate', 'bowl', 'pot', 'pan', 'utensil', 'cutlery', 'mug', 'glass']
    SPORTS = ['ball', 'racket', 'bicycle', 'skateboard', 'weights', 'mat', 'rope']
    OUTDOOR = ['tent', 'backpack', 'sleeping bag', 'water bottle', 'hiking']
    LUXURY = ['premium', 'luxury', 'premium', 'designer', 'high-end', 'exclusive']
    
    # Material keywords
    MATERIALS = {
        'leather': {
            'keywords': ['leather', 'suede', 'calfskin'],
            'description': 'premium leather, detailed grain texture, rich color, perfectly stitched'
        },
        'fabric': {
            'keywords': ['fabric', 'cotton', 'wool', 'linen', 'polyester', 'silk'],
            'description': 'fine fabric weave, textile pattern, smooth finish, natural fibers'
        },
        'metal': {
            'keywords': ['metal', 'steel', 'aluminum', 'chrome', 'stainless', 'iron'],
            'description': 'polished metal, metallic sheen, reflective surfaces, industrial finish'
        },
        'glass': {
            'keywords': ['glass', 'crystal', 'transparent', 'clear'],
            'description': 'crystal clear, perfect transparency, light refraction, pristine'
        },
        'ceramic': {
            'keywords': ['ceramic', 'porcelain', 'pottery'],
            'description': 'smooth ceramic glaze, glossy finish, artistic pattern'
        },
        'plastic': {
            'keywords': ['plastic', 'rubber', 'silicone'],
            'description': 'matte plastic, smooth surfaces, modern aesthetic'
        },
        'wood': {
            'keywords': ['wood', 'wooden', 'timber', 'oak', 'walnut', 'mahogany'],
            'description': 'natural wood grain, warm wood tones, polished finish, detailed grain'
        }
    }
    
    # Category-specific styles
    CATEGORY_STYLES = {
        'electronics': 'sleek modern design, minimalist aesthetic, tech product, futuristic',
        'fashion': 'fashion photography, flatlay style, elegant presentation, styled shot',
        'furniture': 'interior design style, lifestyle photography, modern furniture, comfortable aesthetic',
        'home': 'tabletop photography, food styling, domestic, kitchen aesthetic',
        'sports': 'action-ready, athletic, sports equipment, dynamic posture',
        'luxury': 'high-end luxury, premium presentation, exclusive showcase, affluent aesthetic'
    }
    
    @staticmethod
    def detect_category(description):
        """Detect product category from description"""
        desc_lower = description.lower()
        
        for word in ProductPromptBuilder.LUXURY:
            if word in desc_lower:
                return 'luxury'
        
        for word in ProductPromptBuilder.ELECTRONICS:
            if word in desc_lower:
                return 'electronics'
        
        for word in ProductPromptBuilder.FASHION:
            if word in desc_lower:
                return 'fashion'
        
        for word in ProductPromptBuilder.FURNITURE:
            if word in desc_lower:
                return 'furniture'
        
        for word in ProductPromptBuilder.HOME:
            if word in desc_lower:
                return 'home'
        
        for word in ProductPromptBuilder.SPORTS:
            if word in desc_lower:
                return 'sports'
        
        for word in ProductPromptBuilder.OUTDOOR:
            if word in desc_lower:
                return 'outdoor'
        
        return 'product'
    
    @staticmethod
    def detect_materials(description):
        """Detect materials from product description"""
        desc_lower = description.lower()
        detected = []
        
        for material, data in ProductPromptBuilder.MATERIALS.items():
            for keyword in data['keywords']:
                if keyword in desc_lower:
                    detected.append(data['description'])
                    break
        
        return detected
    
    @staticmethod
    def build_prompt(product_description, include_brand=False, mode='balanced'):
        """
        Build an optimized prompt for product image generation.
        CRITICAL: Product name MUST be at the start for Stable Diffusion to prioritize it.
        
        Parameters:
        - product_description: What product to generate
        - mode: 'balanced' (good product + good photography) or 'product-focused' (MAXIMIZE product accuracy)
        """
        
        if mode == 'product-focused':
            return ProductPromptBuilder.build_product_focused_prompt(product_description)
        
        category = ProductPromptBuilder.detect_category(product_description)
        materials = ProductPromptBuilder.detect_materials(product_description)
        
        # === PRODUCT NAME AT THE TOP - THIS IS THE KEY ===
        prompt_parts = [f"a {product_description}"]
        
        # Add material details RIGHT AFTER product
        if materials:
            # Take only the first material for clarity
            material_detail = materials[0].split(',')[0]  # Get first part
            prompt_parts.append(f"({material_detail})")
        
        # Photography style - SIMPLIFIED
        prompt_parts.append("professional product shot")
        
        # Category-specific styling - MINIMAL
        style = ProductPromptBuilder.CATEGORY_STYLES.get(category, '').split(',')[0]  # First keyword only
        if style:
            prompt_parts.append(style)
        
        # Lighting and environment - CONCISE
        prompt_parts.append("studio lighting, white background")
        
        # Quality - BRIEF
        prompt_parts.append("sharp focus, 8k, trending on artstation")
        
        # Join with commas for Stable Diffusion
        prompt = ", ".join(prompt_parts)
        
        # Keep under 150 chars - Stable Diffusion works better with SHORT prompts
        if len(prompt) > 150:
            # Trim from the end
            prompt = prompt[:150].rsplit(',', 1)[0]
        
        return prompt
    
    @staticmethod
    def build_product_focused_prompt(product_description):
        """
        AGGRESSIVE mode: Maximize probability of getting the RIGHT PRODUCT.
        Sacrifices some photography polish for accuracy.
        """
        # Ultra-simple: just the product + minimal styling
        prompt = f"{product_description}, professional photo, white background, sharp, 8k"
        return prompt
    
    @staticmethod
    def get_negative_prompt():
        """Get negative prompt to avoid unwanted artifacts"""
        return (
            "blurry, low quality, distorted, deformed, amateur, poorly made, "
            "watermark, text, logo, signature, cropped, duplicate, wrong angle, "
            "blurry background, bad focus, out of focus, distortion, "
            "nsfw, nudity, explicit, inappropriate, arm cut off, leg cut off"
        )
    
    @staticmethod
    def build_api_prompt(api_data, mode='cinematic'):
        """
        Build prompt from structured API data for STRIKING social media ads.
        
        Parameters:
        - api_data: dict with keys:
            - service_or_product: What to advertise (e.g., "premium wireless headphones")
            - ideal_market: Target audience (e.g., "tech enthusiasts", "fashion-forward professionals")
            - background_context: (optional) "lifestyle", "nature", "urban", "abstract", "product-only"
            - lighting_style: (optional) "dramatic", "cinematic", "natural", "luxury"
        
        - mode: 'cinematic' (default - very detailed), 'social' (optimized for social media)
        
        Returns: Detailed, visually striking prompt optimized for Instagram/Facebook
        """
        
        product = api_data.get('service_or_product', 'product')
        market = api_data.get('ideal_market', 'general audience')
        background = api_data.get('background_context', 'contextual')
        lighting = api_data.get('lighting_style', 'dramatic')
        
        # ========== DETAILED CINEMATIC PROMPT ==========
        if mode == 'cinematic':
            # Start with product (CRITICAL for SD)
            prompt_parts = [
                f"A visually striking advertisement showcasing {product}",
                f"the {product} prominently displayed in the center, hero shot",
                f"created for {market}, attention-grabbing layout",
                f"{lighting} lighting with realistic reflections, cinematic rendering",
                f"modern design with high contrast colors, professional composition",
                f"suitable for Instagram, Facebook, and social media ads",
            ]
            
            # Add contextual background
            if background.lower() == 'lifestyle':
                prompt_parts.append("lifestyle photography, real-world usage, human context, aspirational")
            elif background.lower() == 'nature':
                prompt_parts.append("natural background, outdoor environment, organic elements, earthy tones")
            elif background.lower() == 'urban':
                prompt_parts.append("urban setting, city backdrop, architectural elements, modern environment")
            elif background.lower() == 'abstract':
                prompt_parts.append("abstract background with complementary patterns, artistic design, dynamic elements")
            else:
                prompt_parts.append("complementary elements and background that fit the product aesthetic")
            
            # Visual emphasis
            prompt_parts.extend([
                "high-resolution, ultra-detailed, 8k quality",
                "trending on behance, artstation aesthetic",
                "professional lighting setup, perfect focus",
                "color graded, premium visual composition"
            ])
            
            prompt = ", ".join(prompt_parts)
            
        # ========== OPTIMIZED FOR SOCIAL MEDIA ==========
        else:  # mode == 'social' or any other value
            prompt_parts = [
                f"{product}",
                f"center stage, hero shot, attention-grabbing",
                f"dramatic lighting, realistic reflections",
                f"high contrast colors, modern design",
                f"professional product photography for {market}",
                f"{background} background with complementary elements",
                f"cinematic rendering, 8k resolution, professional quality"
            ]
            
            prompt = ", ".join(prompt_parts)
        
        return prompt
    
    @staticmethod
    def get_generation_params(quality='high'):
        """
        Get optimized parameters for Stable Diffusion.
        
        Parameters:
        - quality: 'fast' (30 steps), 'high' (50 steps), 'ultra' (75 steps)
        """
        steps_map = {
            'fast': 30,
            'high': 50,
            'ultra': 75
        }
        
        return {
            'num_inference_steps': steps_map.get(quality, 50),
            'guidance_scale': 7.5,       # How much to follow the prompt (7-8.5 is optimal)
            'height': 768,               # Better for product images
            'width': 768,
            'seed': None                 # Set to fixed number for reproducibility
        }


# Test/Example usage
if __name__ == '__main__':
    test_products = [
        "premium wireless noise-canceling headphones",
        "luxury Italian leather backpack",
        "modern coffee table",
        "ceramic mug with hand-painted pattern"
    ]
    
    print("="*80)
    print("PRODUCT PROMPT ENGINEERING")
    print("="*80 + "\n")
    
    for product in test_products:
        print(f"📦 Product: {product}")
        category = ProductPromptBuilder.detect_category(product)
        materials = ProductPromptBuilder.detect_materials(product)
        prompt = ProductPromptBuilder.build_prompt(product)
        negative = ProductPromptBuilder.get_negative_prompt()
        params = ProductPromptBuilder.get_generation_params()
        
        print(f"   📂 Category: {category}")
        print(f"   🎨 Materials: {materials if materials else 'None detected'}")
        print(f"   📝 Prompt: {prompt[:100]}...")
        print(f"   ❌ Negative: {negative[:80]}...")
        print(f"   ⚙️  Params: steps={params['num_inference_steps']}, scale={params['guidance_scale']}\n")
