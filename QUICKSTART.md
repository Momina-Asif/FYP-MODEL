# 🚀 Quick Start Guide

Get your poster generator up and running in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- CUDA 11.8+ (for GPU acceleration - optional but recommended)
- 8GB RAM minimum (16GB+ recommended for Stable Diffusion)
- 6GB GPU VRAM (for NVIDIA GPUs)

## Step 1: Clone/Setup Project

```bash
# Navigate to your project
cd FYP-MODEL

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch with CUDA support
- Stable Diffusion (diffusers)
- FastAPI (for API server)
- Pillow (image processing)
- And more...

**Installation time**: 5-15 minutes depending on your internet

## Step 3: Try It Out

### Option A: Simple Python Script (Fastest)

```python
from core.poster_generator import generate_poster_simple

poster = generate_poster_simple(
    cta="Join Now",
    target_audience="Tech enthusiasts",
    product_description="Tech Conference 2026",
    output_path="my_first_poster.png"
)
print("✅ Poster saved as my_first_poster.png")
```

Save this as `test.py` and run:
```bash
python test.py
```

### Option B: Command Line

```bash
python cli.py \
  --cta "Join Now" \
  --audience "Tech enthusiasts and professionals" \
  --description "Annual Technology Conference 2026" \
  --style modern \
  --output conference_poster.png
```

### Option C: API Server

**Terminal 1 - Start the server:**
```bash
python -m api.server
```

You'll see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     ✅ API ready to serve requests
```

**Terminal 2 - Make a request:**
```bash
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "cta": "Join Now",
    "target_audience": "Tech enthusiasts",
    "product_description": "Tech Summit 2026",
    "style": "modern"
  }' \
  --output poster.png
```

Or visit: http://localhost:8000/docs for interactive API documentation

## What Happens During Generation?

1. **Models Load** (~30-60 seconds on first run)
   - Layout model loads
   - Stable Diffusion model downloads (~4GB on first run)
   - Text encoder initializes

2. **Generation Pipeline** (~2-5 minutes)
   - Layout is generated based on your content
   - Background image is created using Stable Diffusion
   - Text elements are rendered onto background
   - Final poster is saved

3. **Output** 
   - Poster image (PNG/JPG)
   - Metadata (element positions, types, text)

## Configuration

Edit `config.py` to customize:

```python
IMAGE_SIZE = 1024           # Poster size (smaller = faster)
NUM_BOXES = 10             # Layout elements
DEVICE = "cuda"            # Use CUDA for faster generation
```

For slower systems, try:
```python
IMAGE_SIZE = 512           # Will be ~4x faster
DEVICE = "cpu"             # If no GPU
```

## Common Issues & Solutions

### Issue: "CUDA out of memory"
```python
# In config.py:
IMAGE_SIZE = 512  # Reduce size
# Or use CPU:
DEVICE = "cpu"
```

### Issue: "Module not found"
```bash
# Make sure you're in the virtual environment
# Windows:
venv\Scripts\activate
# Install again:
pip install -r requirements.txt
```

### Issue: Slow generation
- First run downloads Stable Diffusion (~4GB) - this takes time
- GPU acceleration (CUDA) is much faster than CPU
- Reduce `IMAGE_SIZE` in config.py

### Issue: "Models not found"
Make sure you have trained models:
- `layout_model.pth` - in project root
- `text_model.pth` - (optional)

If missing, the system will use random initialization

## Next Steps

1. **Explore Examples**: Check `examples/` folder for more usage patterns
2. **Customize Styles**: Try different styles: modern, vibrant, elegant, bold
3. **Batch Processing**: Generate multiple posters efficiently
4. **Deploy**: Use Docker for production deployment
5. **Integrate**: Add to your Django/Flask/FastAPI app

## Example Code Snippets

### Generate 5 Different Posters

```python
from core.poster_generator import PosterGenerator

gen = PosterGenerator()

configs = [
    {"cta": "Buy Now", "product_description": "Summer Sale 2026"},
    {"cta": "Register", "product_description": "ML Course"},
    {"cta": "Learn More", "product_description": "New Product"},
]

for idx, config in enumerate(configs, 1):
    poster, metadata = gen.generate(
        cta=config["cta"],
        target_audience="Everyone",
        product_description=config["product_description"],
        style="modern",
        output_path=f"poster_{idx}.png"
    )
    print(f"✅ Poster {idx} created")
```

### Make API Call from Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/generate",
    json={
        "cta": "Join Now",
        "target_audience": "Tech professionals",
        "product_description": "Tech Summit",
        "style": "modern"
    }
)

with open("api_poster.png", "wb") as f:
    f.write(response.content)
print("✅ Poster downloaded")
```

## Benchmarks

On NVIDIA RTX 3090:
- **First generation**: ~3-5 minutes (model loading + generation)
- **Subsequent generations**: ~1-2 minutes
- **CPU (no GPU)**: ~15-30 minutes

## Resources

- 📖 [Full Documentation](README_FRAMEWORK.md)
- 🎨 [API Documentation](README_FRAMEWORK.md#-api-documentation)
- 💻 [Examples](examples/)
- 🐳 [Docker Setup](Dockerfile)

## Get Help

1. Check full README in [README_FRAMEWORK.md](README_FRAMEWORK.md)
2. Look at [examples/](examples/) folder
3. Review [config.py](config.py) for configuration options
4. Check API docs at http://localhost:8000/docs (when server running)

---

**Happy Poster Generating! 🎉**

Questions? Issues? Check the full documentation or examples folder!
