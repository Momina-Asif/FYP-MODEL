"""
Example 2: Advanced Python API Usage
Generate multiple posters with different styles and content
"""

from core.poster_generator import PosterGenerator
import os

def generate_multiple_posters():
    """Generate posters for different products/events"""
    
    # Initialize generator once
    generator = PosterGenerator()
    
    # Create output directory
    os.makedirs("posters_batch", exist_ok=True)
    
    # Define different poster configs
    configs = [
        {
            "cta": "Buy Now",
            "target_audience": "Fashion enthusiasts and trend followers",
            "product_description": "Summer Collection 2026 - Exclusive Fashion Line",
            "style": "vibrant",
        },
        {
            "cta": "Register Today",
            "target_audience": "Students and professionals",
            "product_description": "Machine Learning Bootcamp - Master AI and Deep Learning",
            "style": "modern",
        },
        {
            "cta": "Get Yours",
            "target_audience": "Tech gadget enthusiasts",
            "product_description": "Latest Smartphone Launch - Revolutionary Features",
            "style": "bold",
        },
        {
            "cta": "Explore Now",
            "target_audience": "Art and design lovers",
            "product_description": "Digital Art Exhibition - Contemporary Masterpieces",
            "style": "artistic",
        },
    ]
    
    # Generate posters
    for idx, config in enumerate(configs, 1):
        print(f"\n🎨 Generating poster {idx}/{len(configs)}...")
        print(f"   Product: {config['product_description']}")
        
        output_path = f"posters_batch/poster_{idx}.png"
        
        poster, metadata = generator.generate(
            cta=config["cta"],
            target_audience=config["target_audience"],
            product_description=config["product_description"],
            style=config["style"],
            output_path=output_path
        )
        
        print(f"   ✅ Saved to {output_path}")
        print(f"   📊 Elements: {metadata['num_boxes']}")


if __name__ == "__main__":
    print("🚀 Starting batch poster generation...")
    generate_multiple_posters()
    print("\n✅ All posters generated successfully!")
