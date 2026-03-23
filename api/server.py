"""
FastAPI server for Poster Generation
Exposes endpoints to generate posters via HTTP API calls
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager
import sys

from complete_pipeline import generate_poster_text, generate_product_image, create_professional_poster
from prompt_engineering import ProductPromptBuilder
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# No need for global poster generator - complete_pipeline handles it internally


class PosterRequest(BaseModel):
    """Request model for poster generation"""
    cta: str = "Click Here"
    target_audience: str = "Everyone"
    product_description: str = "Amazing Product"
    product: Optional[str] = None  # Accept "product" as alias
    model: str = "base"  # Support model selection
    style: str = "modern"
    format: str = "png"
    
    def __init__(self, **data):
        # Support "product" as alias for "product_description"
        if 'product' in data and 'product_description' not in data:
            data['product_description'] = data.pop('product')
        super().__init__(**data)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "product": "woman shirt",
                "model": "base",
                "cta": "Shop Now",
                "target_audience": "Fashion enthusiasts",
                "style": "modern",
                "format": "png"
            }
        }
    }


class PosterResponse(BaseModel):
    """Response model for poster generation"""
    success: bool
    message: str
    metadata: Optional[dict] = None


class SocialMediaAdRequest(BaseModel):
    """Request model for striking social media ad generation with structured data"""
    service_or_product: str
    ideal_market: str
    background_context: Optional[str] = "contextual"  # lifestyle, nature, urban, abstract, product-only
    lighting_style: Optional[str] = "dramatic"  # dramatic, cinematic, natural, luxury
    mode: Optional[str] = "cinematic"  # cinematic (detailed) or social (optimized for social media)
    quality: Optional[str] = "high"  # fast, high, ultra
    format: str = "png"
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "service_or_product": "premium wireless noise-canceling headphones",
                "ideal_market": "tech enthusiasts and professionals",
                "background_context": "lifestyle",
                "lighting_style": "dramatic",
                "mode": "cinematic",
                "quality": "high",
                "format": "png"
            }
        }
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("🚀 Starting Poster Generator API...")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    logger.info(f"Using device: {device}")
    
    logger.info("✅ API ready to serve requests (using complete_pipeline)")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Poster Generator API...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Poster Generator API",
    description="Generate professional posters using AI with Stable Diffusion backgrounds",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Poster & Social Media Ad Generator API",
        "version": "2.0.0",
        "endpoints": {
            "professional_poster": "/api/generate-professional-poster (POST) - ⭐ FULL PIPELINE: Trained text + HF API image + Professional layout",
            "basic_poster": "/api/generate (POST) - Basic poster with trained text + image",
            "social_media_ads": "/api/generate-social-media-ad (POST) - Image only (no text)",
            "debug_prompts": "/api/generate-social-media-ad-metadata (POST) - See generated prompts",
            "health": "/health (GET)",
            "styles": "/api/styles (GET)",
            "docs": "/docs (GET)"
        },
        "note": "Use /api/generate-professional-poster for complete poster with all features!"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Poster Generator",
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/api/generate")
async def generate_poster(request: PosterRequest):
    """
    Generate a COMPLETE professional poster using the full pipeline:
    1. AI-generated marketing text from trained model
    2. AI-generated product image from Stable Diffusion
    3. Professional poster layout combining both
    
    **Parameters:**
    - **product_description**: Description of the product (e.g., "premium wireless headphones")
    - **cta**: Call-to-action text (e.g., "Join Now", "Buy Today")
    - **target_audience**: Description of target audience (optional - for context)
    - **style**: Visual style (optional - currently not used in complete_pipeline)
    - **format**: Output format (png, jpg)
    
    **Returns:**
    - Generated professional poster image with AI text and AI product image
    
    **Example:**
    ```json
    {
        "product_description": "premium wireless noise-canceling headphones",
        "cta": "Shop Now",
        "target_audience": "Tech enthusiasts and professionals",
        "style": "modern",
        "format": "png"
    }
    ```
    
    **⭐ This endpoint uses:**
    - ✅ Trained seq2seq model for marketing text generation
    - ✅ Stable Diffusion for AI product images
    - ✅ Professional poster layout with adaptive colors
    """
    try:
        logger.info(f"📨 Received complete poster request: {request.product_description}")
        
        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Generate text with trained model
        logger.info("  🤖 Step 1: Generating marketing text from trained model...")
        text_dict = generate_poster_text(request.product_description, use_base=(request.model == "base"))
        logger.info(f"     ✅ Title: {text_dict['title']}")
        logger.info(f"     ✅ Subtitle: {text_dict['subtitle']}")
        logger.info(f"     ✅ Description: {text_dict['description']}")
        logger.info(f"     ✅ CTA: {text_dict['cta']}")
        
        # Step 2: Generate product image with Stable Diffusion
        logger.info("  🎨 Step 2: Generating product image with Stable Diffusion...")
        image_path = os.path.join(output_dir, f"product_{timestamp}.png")
        generate_product_image(request.product_description, image_path)
        logger.info(f"     ✅ Product image: {image_path}")
        
        # Step 3: Create professional poster
        logger.info("  📄 Step 3: Creating professional poster layout...")
        poster_path = os.path.join(output_dir, f"poster_{timestamp}.{request.format}")
        create_professional_poster(text_dict, image_path, poster_path)
        logger.info(f"✅ Complete poster generated: {poster_path}")
        
        # Return image file
        return FileResponse(
            poster_path,
            media_type=f"image/{request.format}",
            filename=f"poster_{timestamp}.{request.format}"
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating complete poster: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-professional-poster")
async def generate_professional_poster(request: SocialMediaAdRequest):
    """
    Generate COMPLETE professional poster with:
    1. Trained model for marketing text (uses product description)
    2. Stable Diffusion 3 via HF API for product image
    3. Professional layout with text overlay
    
    ⭐ **THIS IS THE ENDPOINT YOU WANT!**
    
    **Parameters:**
    - **service_or_product**: Product/service to advertise (e.g., "premium wireless headphones")
    - **ideal_market**: Target audience (e.g., "tech enthusiasts")
    - **background_context**: Type of background (lifestyle, nature, urban, abstract, product-only)
    - **lighting_style**: Lighting (dramatic, cinematic, natural, luxury)
    - **mode**: Prompt mode (cinematic, social)
    - **quality**: Quality (fast, high, ultra)
    - **format**: Output format (png, jpg)
    
    **Example:**
    ```json
    {
        "service_or_product": "premium wireless noise-canceling headphones",
        "ideal_market": "tech enthusiasts and professionals",
        "background_context": "lifestyle",
        "lighting_style": "dramatic",
        "mode": "cinematic",
        "quality": "high",
        "format": "png"
    }
    ```
    
    **Returns:** Professional poster image with AI text + AI image
    """
    try:
        logger.info(f"🎯 Professional Poster Request: {request.service_or_product} for {request.ideal_market}")
        
        # Use service_or_product as product_description
        product_description = request.service_or_product
        
        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Generate text with trained model
        logger.info("  🤖 Step 1: Generating marketing text from trained model...")
        text_dict = generate_poster_text(product_description)
        logger.info(f"     ✅ Title: {text_dict['title']}")
        logger.info(f"     ✅ Subtitle: {text_dict['subtitle'][:40]}...")
        logger.info(f"     ✅ CTA: {text_dict['cta']}")
        
        # Step 2: Generate product image with HF API
        logger.info("  🎨 Step 2: Generating product image via HF API...")
        image_path = os.path.join(output_dir, f"product_{timestamp}.png")
        generate_product_image(product_description, image_path)
        logger.info(f"     ✅ Product image: {image_path}")
        
        # Step 3: Create professional poster
        logger.info("  📄 Step 3: Creating professional poster layout...")
        poster_path = os.path.join(output_dir, f"poster_{timestamp}.{request.format}")
        create_professional_poster(text_dict, image_path, poster_path)
        logger.info(f"✅ Professional poster generated: {poster_path}")
        
        # Return image file
        return FileResponse(
            poster_path,
            media_type=f"image/{request.format}",
            filename=f"poster_{timestamp}.{request.format}"
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating professional poster: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-and-metadata")
async def generate_poster_with_metadata(request: PosterRequest):
    """
    Generate a poster and return both image and metadata as JSON.
    Useful for receiving structured data about the generated poster.
    """
    try:
        if not poster_gen:
            raise HTTPException(status_code=500, detail="Generator not initialized")
        
        logger.info(f"📨 Received request (with metadata): {request.product_description}")
        
        # Generate poster (in memory, no file save)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"poster_{timestamp}.png")
        
        poster_image, metadata = poster_gen.generate(
            cta=request.cta,
            target_audience=request.target_audience,
            product_description=request.product_description,
            style=request.style,
            output_path=output_path
        )
        
        return JSONResponse({
            "success": True,
            "message": "Poster generated successfully",
            "image_path": output_path,
            "metadata": metadata,
            "request": request.model_dump()
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/styles")
async def get_available_styles():
    """Get list of available poster styles"""
    styles = [
        "modern",
        "minimalist",
        "vibrant",
        "elegant",
        "bold",
        "playful",
        "professional",
        "artistic"
    ]
    return {"styles": styles}


@app.post("/api/generate-social-media-ad")
async def generate_social_media_ad(request: SocialMediaAdRequest):
    """
    Generate STRIKING social media ads with advanced prompt engineering.
    ⭐ NEW ENDPOINT - Use this for visually striking ads with proper backgrounds!
    
    **Parameters:**
    - **service_or_product**: Product/service to advertise (e.g., "premium wireless headphones")
    - **ideal_market**: Target audience (e.g., "tech enthusiasts", "fitness lovers")
    - **background_context**: Type of background
      - `lifestyle` - Real-world usage, aspirational
      - `nature` - Outdoor, organic elements
      - `urban` - City setting, modern architecture
      - `abstract` - Artistic patterns, dynamic design
      - `product-only` - Clean white background
    - **lighting_style**: Lighting approach
      - `dramatic` - High contrast, cinematic (DEFAULT)
      - `cinematic` - Movie-like quality
      - `natural` - Soft, authentic
      - `luxury` - Premium, sophisticated
    - **mode**: Prompt detail level
      - `cinematic` - Very detailed, striking visuals (DEFAULT)
      - `social` - Optimized for social media platforms
    - **quality**: Generation quality
      - `fast` - 30 steps, quick generation
      - `high` - 50 steps, good balance (DEFAULT)
      - `ultra` - 75 steps, highest quality
    
    **Example:**
    ```json
    {
        "service_or_product": "premium wireless noise-canceling headphones",
        "ideal_market": "tech professionals and music lovers",
        "background_context": "lifestyle",
        "lighting_style": "dramatic",
        "mode": "cinematic",
        "quality": "high",
        "format": "png"
    }
    ```
    
    **Returns:** Generated social media ad image
    """
    try:
        if not poster_gen:
            raise HTTPException(status_code=500, detail="Generator not initialized")
        
        logger.info(f"📱 Social Media Ad Request: {request.service_or_product} for {request.ideal_market}")
        
        # Build API data dict
        api_data = {
            "service_or_product": request.service_or_product,
            "ideal_market": request.ideal_market,
            "background_context": request.background_context,
            "lighting_style": request.lighting_style
        }
        
        # Generate optimized prompt using new API prompt builder
        prompt = ProductPromptBuilder.build_api_prompt(api_data, mode=request.mode)
        negative_prompt = ProductPromptBuilder.get_negative_prompt()
        params = ProductPromptBuilder.get_generation_params(quality=request.quality)
        
        logger.info(f"🎨 Generated Prompt ({len(prompt)} chars): {prompt[:100]}...")
        
        # Generate timestamp and path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"social_ad_{timestamp}.{request.format}")
        
        # Try to generate with Stable Diffusion
        try:
            from diffusers import StableDiffusionPipeline
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            pipe = pipe.to(device)
            
            # Generate image
            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                **params
            ).images[0]
            
            image.save(output_path)
            
            logger.info(f"✅ Social media ad generated: {output_path}")
            
            return FileResponse(
                output_path,
                media_type=f"image/{request.format}",
                filename=f"social_ad_{timestamp}.{request.format}",
                headers={
                    "X-Prompt": prompt[:200],
                    "X-Product": request.service_or_product,
                    "X-Market": request.ideal_market
                }
            )
            
        except Exception as e:
            logger.warning(f"⚠️  Stable Diffusion generation failed: {str(e)}, using placeholder")
            # Fall back to placeholder (existing poster generator)
            return JSONResponse({
                "success": False,
                "message": f"Image generation failed: {str(e)}",
                "prompt_generated": prompt,
                "error_details": str(e)
            }, status_code=500)
        
    except Exception as e:
        logger.error(f"❌ Error generating social media ad: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-social-media-ad-metadata")
async def generate_social_media_ad_metadata(request: SocialMediaAdRequest):
    """
    Generate social media ad and return both image path + prompt metadata.
    Useful for debugging and understanding what prompts produce what images.
    """
    try:
        # Build API data
        api_data = {
            "service_or_product": request.service_or_product,
            "ideal_market": request.ideal_market,
            "background_context": request.background_context,
            "lighting_style": request.lighting_style
        }
        
        # Generate prompt
        prompt = ProductPromptBuilder.build_api_prompt(api_data, mode=request.mode)
        negative_prompt = ProductPromptBuilder.get_negative_prompt()
        params = ProductPromptBuilder.get_generation_params(quality=request.quality)
        
        logger.info(f"📱 Social Ad with Metadata: {request.service_or_product}")
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"social_ad_{timestamp}.{request.format}")
        
        return JSONResponse({
            "success": True,
            "message": "Prompt generated successfully",
            "image_path": output_path,
            "metadata": {
                "prompt": prompt,
                "prompt_length": len(prompt),
                "negative_prompt": negative_prompt,
                "generation_params": params,
                "api_input": api_data
            },
            "request": request.model_dump()
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
