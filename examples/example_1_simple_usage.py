"""
Example 1: Simple Python API Usage
Generate a poster using Python directly without API server
"""

from core.poster_generator import generate_poster_simple

# Example 1: Simple usage
if __name__ == "__main__":
    poster = generate_poster_simple(
        cta="Join Now",
        target_audience="Tech enthusiasts and professionals",
        product_description="Annual Technology Conference 2026 - Innovation Summit",
        style="modern",
        output_path="example_poster_1.png"
    )
    print("✅ Poster saved as example_poster_1.png")

    # You can now use the image further or display it
    # poster.show()  # Uncomment to display
