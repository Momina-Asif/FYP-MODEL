#!/usr/bin/env python
"""
Replace weak Seq2Seq model with Claude API for context-aware text generation.
"""

import os
import json
from anthropic import Anthropic

def generate_poster_text_with_llm(product_description):
    """
    Generate marketing text using Anthropic Claude API.
    This ACTUALLY uses the product description, unlike the broken seq2seq model.
    """
    
    client = Anthropic()
    
    prompt = f"""You are an expert marketing copywriter. Generate compelling poster text for: {product_description}

Return ONLY JSON (no markdown, no explanation):
{{
  "title": "Short catchy headline (3-5 words)",
  "subtitle": "Subheading emphasizing key benefit (1 line)",
  "description": "Brief description what makes it special (2 lines max)",
  "cta": "Call to action button text"
}}"""
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {{"role": "user", "content": prompt}}
        ]
    )
    
    response_text = message.content[0].text.strip()
    
    # Extract JSON from response
    try:
        # Try direct parse first
        result = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON if wrapped in markdown
        if "```" in response_text:
            json_str = response_text.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            result = json.loads(json_str)
        else:
            raise ValueError(f"Could not parse LLM response: {response_text}")
    
    return {{
        'title': result.get('title', 'Premium Product'),
        'subtitle': result.get('subtitle', 'Exceptional quality'),
        'description': result.get('description', 'Discover excellence'),
        'cta': result.get('cta', 'Shop Now')
    }}


# Test it
if __name__ == "__main__":
    test_products = [
        "premium wireless noise-canceling headphones",
        "luxury leather travel backpack",
        "modern smartwatch with fitness tracking"
    ]
    
    for product in test_products:
        print(f"\n📦 Product: {product}")
        print("-" * 60)
        text = generate_poster_text_with_llm(product)
        for key, value in text.items():
            print(f"  {key}: {value}")
