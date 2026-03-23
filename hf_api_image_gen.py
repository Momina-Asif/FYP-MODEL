"""
Hugging Face API Image Generation
Uses your backend's HF API approach for more reliable production image generation
"""

import os
import requests
from io import BytesIO
from PIL import Image


def generate_product_image_hf_api(product_description, output_path, hf_api_key=None):
    """
    Generate product image using Hugging Face API (from your backend approach).
    More reliable than local Stable Diffusion for production use.
    
    Args:
        product_description: What to generate
        output_path: Where to save the image
        hf_api_key: HF API key (defaults to HF_API_KEY env var)
    
    Returns: output_path if successful, None otherwise
    """
    
    if not hf_api_key:
        hf_api_key = os.getenv("HF_API_KEY")
    
    if not hf_api_key:
        print("   WARNING: HF_API_KEY not set, skipping HF API generation")
        return None
    
    try:
        # Create optimized prompt
        from prompt_engineering import ProductPromptBuilder
        prompt = ProductPromptBuilder.build_prompt(product_description, mode='product-focused')
        negative_prompt = ProductPromptBuilder.get_negative_prompt()
        
        model_url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-3-medium-diffusers"
        headers = {
            "Authorization": f"Bearer {hf_api_key}",
            "Content-Type": "application/json"
        }
        payload = {"inputs": prompt}
        
        print(f"   Generating via HF API: {prompt[:60]}...")
        response = requests.post(model_url, headers=headers, json=payload, timeout=120)
        
        if response.status_code != 200:
            print(f"   ERROR HF API: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return None
        
        # Save image from response
        image_data = response.content
        if not image_data:
            print("   ERROR: No image data returned from HF API")
            return None
            
        image = Image.open(BytesIO(image_data))
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        image.save(output_path)
        print(f"   SUCCESS: Image saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"   ERROR: HF API generation failed: {str(e)}")
        return None


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Example: Using HF API for image generation\n")
    
    # Make sure your HF_API_KEY is set in environment
    # export HF_API_KEY="your_key_here"
    
    result = generate_product_image_hf_api(
        "premium wireless noise-canceling headphones",
        "output/hf_api_test.png"
    )
    
    if result:
        print(f"\n✅ Image generated successfully: {result}")
    else:
        print("\n❌ Image generation failed")
