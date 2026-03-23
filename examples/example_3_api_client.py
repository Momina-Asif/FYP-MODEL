"""
Example 3: API Server Usage via HTTP Requests
Shows how to call the poster generator API using requests
"""

import requests
import json
from typing import Optional
import os


class PosterGeneratorClient:
    """
    Client for interacting with Poster Generator API
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
    
    def check_health(self) -> bool:
        """Check if API server is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def generate_poster(
        self,
        cta: str,
        target_audience: str,
        product_description: str,
        style: str = "modern",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a poster via API
        
        Args:
            cta: Call-to-action text
            target_audience: Target audience description
            product_description: Product description
            style: Visual style
            save_path: Path to save the poster (optional)
            
        Returns:
            Path to saved poster or None
        """
        
        payload = {
            "cta": cta,
            "target_audience": target_audience,
            "product_description": product_description,
            "style": style,
            "format": "png"
        }
        
        try:
            print(f"📡 Sending request to API: {self.base_url}/api/generate")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300  # 5 minutes timeout for image generation
            )
            
            if response.status_code == 200:
                if save_path is None:
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = f"api_poster_{timestamp}.png"
                
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ Poster saved to: {save_path}")
                return save_path
            else:
                error = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                print(f"❌ API Error: {error}")
                return None
                
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
            return None
    
    def generate_poster_with_metadata(
        self,
        cta: str,
        target_audience: str,
        product_description: str,
        style: str = "modern"
    ) -> Optional[dict]:
        """
        Generate a poster and get metadata
        
        Args:
            cta: Call-to-action text
            target_audience: Target audience description
            product_description: Product description
            style: Visual style
            
        Returns:
            Dictionary with image path and metadata
        """
        
        payload = {
            "cta": cta,
            "target_audience": target_audience,
            "product_description": product_description,
            "style": style,
            "format": "png"
        }
        
        try:
            print(f"📡 Sending request to API: {self.base_url}/api/generate-and-metadata")
            response = requests.post(
                f"{self.base_url}/api/generate-and-metadata",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Poster generated")
                print(f"   Image: {data['image_path']}")
                print(f"   Elements: {data['metadata']['num_elements']}")
                return data
            else:
                print(f"❌ API Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
            return None
    
    def get_styles(self) -> list:
        """Get available poster styles"""
        try:
            response = requests.get(f"{self.base_url}/api/styles")
            if response.status_code == 200:
                return response.json()["styles"]
        except:
            pass
        return []


def main():
    """
    Example usage of the API client
    Make sure the API server is running: python api/server.py
    """
    
    # Initialize client
    client = PosterGeneratorClient(base_url="http://localhost:8000")
    
    # Check if API is available
    print("🔍 Checking API availability...")
    if not client.check_health():
        print("❌ API server is not running!")
        print("   Start it with: python api/server.py")
        return
    
    print("✅ API server is running!\n")
    
    # Get available styles
    print("📋 Available styles:")
    styles = client.get_styles()
    for style in styles:
        print(f"   - {style}")
    print()
    
    # Example 1: Generate a single poster
    print("=" * 50)
    print("Example 1: Tech Conference Poster")
    print("=" * 50)
    client.generate_poster(
        cta="Register Now",
        target_audience="Software developers and tech enthusiasts",
        product_description="International Tech Summit 2026 - AI & Innovation",
        style="modern",
        save_path="conference_poster.png"
    )
    
    # Example 2: Generate with metadata
    print("\n" + "=" * 50)
    print("Example 2: Product Launch Poster (with metadata)")
    print("=" * 50)
    result = client.generate_poster_with_metadata(
        cta="Pre-Order Now",
        target_audience="Tech gadget enthusiasts and early adopters",
        product_description="Next Generation Smartphone - Ultimate Technology",
        style="bold"
    )
    
    if result:
        print("\n📊 Metadata:")
        print(json.dumps(result["metadata"], indent=2))
    
    # Example 3: Batch generation
    print("\n" + "=" * 50)
    print("Example 3: Batch Generation")
    print("=" * 50)
    
    batch_configs = [
        {
            "cta": "Shop Now",
            "target_audience": "Fashion enthusiasts",
            "product_description": "Spring Collection 2026",
            "style": "vibrant",
            "file": "fashion_poster.png"
        },
        {
            "cta": "Join Us",
            "target_audience": "Sports enthusiasts",
            "product_description": "Marathon Championship 2026",
            "style": "bold",
            "file": "sports_poster.png"
        }
    ]
    
    for idx, config in enumerate(batch_configs, 1):
        print(f"\n📨 Generating poster {idx}/{len(batch_configs)}...")
        client.generate_poster(
            cta=config["cta"],
            target_audience=config["target_audience"],
            product_description=config["product_description"],
            style=config["style"],
            save_path=config["file"]
        )
    
    print("\n✅ All examples completed!")


if __name__ == "__main__":
    main()
