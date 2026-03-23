#!/usr/bin/env python
"""
Command-line interface for Poster Generator
Usage:
    python cli.py --cta "Join Now" --audience "Tech enthusiasts" \\
                  --description "Tech Conference 2026" --style modern --output poster.png
"""

import argparse
import sys
from core.poster_generator import PosterGenerator
import logging


def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate professional posters using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple poster
  python cli.py --cta "Join Now" --audience "Tech professionals" \\
                --description "Tech Summit 2026"

  # With custom style
  python cli.py --cta "Buy Now" --audience "Fashion enthusiasts" \\
                --description "Summer Collection" --style vibrant

  # Save to specific path
  python cli.py --cta "Register" --audience "Students" \\
                --description "ML Bootcamp" --output my_poster.png

  # With verbose logging
  python cli.py --cta "Join" --audience "Everyone" \\
                --description "Event 2026" -v
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--cta',
        type=str,
        required=True,
        help='Call-to-action text (e.g., "Join Now", "Buy Today")'
    )
    parser.add_argument(
        '--audience',
        type=str,
        required=True,
        help='Description of target audience'
    )
    parser.add_argument(
        '--description',
        type=str,
        required=True,
        help='Description of product/event'
    )
    
    # Optional arguments
    parser.add_argument(
        '--style',
        type=str,
        choices=['modern', 'minimalist', 'vibrant', 'elegant', 'bold', 'playful', 'professional', 'artistic'],
        default='modern',
        help='Visual style (default: modern)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: auto-generated)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting Poster Generator CLI")
        logger.info(f"Parameters:")
        logger.info(f"  CTA: {args.cta}")
        logger.info(f"  Audience: {args.audience}")
        logger.info(f"  Description: {args.description}")
        logger.info(f"  Style: {args.style}")
        
        # Initialize generator
        logger.info("Loading models...")
        generator = PosterGenerator()
        
        # Generate poster
        logger.info("Generating poster...")
        poster, metadata = generator.generate(
            cta=args.cta,
            target_audience=args.audience,
            product_description=args.description,
            style=args.style,
            output_path=args.output
        )
        
        logger.info("✅ Poster generated successfully!")
        logger.info(f"Metadata: {metadata['num_boxes']} elements included")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
