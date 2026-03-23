# 🎨 AI Poster Generation Framework

A complete end-to-end framework for generating professional posters using:
- **Stable Diffusion** for background image generation
- **Custom Layout Model** for intelligent element placement
- **Custom Text Model** for content generation
- **FastAPI** for easy API access

## 🌟 Features

✅ **End-to-End Pipeline**: Generate complete posters from structured content  
✅ **Stable Diffusion Integration**: Realistic, AI-generated backgrounds  
✅ **Customizable Layouts**: Intelligent element positioning  
✅ **Multiple Input Formats**: Python API, HTTP API, or CLI  
✅ **Batch Processing**: Generate multiple posters efficiently  
✅ **Metadata Output**: Get detailed information about generated posters  

## 📋 Quick Start

### 1. Installation

```bash
# Clone/navigate to the project directory
cd FYP-MODEL

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage (Python API)

```python
from core.poster_generator import generate_poster_simple

# Generate a poster
poster = generate_poster_simple(
    cta="Join Now",
    target_audience="Tech enthusiasts and professionals",
    product_description="Annual Technology Conference 2026",
    style="modern",
    output_path="my_poster.png"
)
```

### 3. Start API Server

```bash
# Start the FastAPI server
python -m api.server

# API will be available at: http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### 4. Make API Calls

```python
import requests

response = requests.post(
    "http://localhost:8000/api/generate",
    json={
        "cta": "Join Now",
        "target_audience": "Tech enthusiasts",
        "product_description": "Tech Conference 2026",
        "style": "modern"
    }
)

with open("poster.png", "wb") as f:
    f.write(response.content)
```

## 📚 Architecture

```
FYP-MODEL/
├── core/
│   ├── __init__.py
│   └── poster_generator.py      # Main orchestrator
│
├── api/
│   ├── __init__.py
│   └── server.py                # FastAPI server
│
├── models/
│   ├── layout_model.py          # Layout generation
│   ├── text_to_layout.py        # Text-to-layout conversion
│   ├── sd_base.py               # Stable Diffusion loading
│   └── ...other models
│
├── utils/
│   ├── content_generator_v2.py  # Enhanced content generation
│   ├── renderer.py              # Rendering utilities
│   └── ...other utilities
│
├── examples/
│   ├── example_1_simple_usage.py
│   ├── example_2_batch_generation.py
│   └── example_3_api_client.py
│
├── config.py                    # Configuration
└── requirements.txt             # Dependencies
```

## 🚀 Usage Modes

### Mode 1: Simple Python Function

```python
from core.poster_generator import generate_poster_simple

poster = generate_poster_simple(
    cta="Buy Now",
    target_audience="Fashion enthusiasts",
    product_description="Summer Collection 2026",
    style="vibrant",
    output_path="summer_poster.png"
)
```

### Mode 2: Advanced Python API

```python
from core.poster_generator import PosterGenerator

generator = PosterGenerator()

poster, metadata = generator.generate(
    cta="Register Today",
    target_audience="Students and professionals",
    product_description="Machine Learning Bootcamp",
    style="modern",
    output_path="ml_poster.png"
)

print(metadata)  # Get detailed information about the poster
```

### Mode 3: HTTP API

**Start Server:**
```bash
python -m api.server
```

**Generate Poster:**
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

**Get Available Styles:**
```bash
curl "http://localhost:8000/api/styles"
```

**Check Health:**
```bash
curl "http://localhost:8000/health"
```

## 📖 API Documentation

### Endpoint: `/api/generate`

Generates a poster and returns the image file.

**Request:**
```json
{
  "cta": "Join Now",
  "target_audience": "Tech enthusiasts and professionals",
  "product_description": "Annual Technology Conference 2026",
  "style": "modern",
  "format": "png"
}
```

**Parameters:**
- `cta` (string): Call-to-action text (e.g., "Join Now", "Buy Today")
- `target_audience` (string): Description of target audience
- `product_description` (string): Description of product/event
- `style` (string): Visual style - modern, minimalist, vibrant, elegant, bold, playful, professional, artistic
- `format` (string): Output format - png (recommended) or jpg

**Response:** Binary image file (PNG/JPG)

### Endpoint: `/api/generate-and-metadata`

Generates a poster and returns both image path and metadata.

**Response:**
```json
{
  "success": true,
  "message": "Poster generated successfully",
  "image_path": "/output/poster_20240101_120000.png",
  "metadata": {
    "elements": [
      {
        "type": "title",
        "text": "TECH CONFERENCE 2026",
        "box": [0.1, 0.1, 0.8, 0.2],
        "position_px": [102, 102, 820, 204]
      }
    ],
    "num_boxes": 10
  }
}
```

### Endpoint: `/api/styles`

Returns list of available poster styles.

**Response:**
```json
{
  "styles": ["modern", "minimalist", "vibrant", "elegant", "bold", "playful", "professional", "artistic"]
}
```

## 🎯 Available Styles

- **modern**: Clean, contemporary design
- **minimalist**: Simple, elegant with lots of whitespace
- **vibrant**: Colorful, energetic design
- **elegant**: Sophisticated, premium look
- **bold**: Strong, impactful design
- **playful**: Fun, creative design
- **professional**: Business-oriented design
- **artistic**: Creative, artistic design

## 🔧 Configuration

Edit `config.py` to modify:

```python
IMG_SIZE = 256                  # Image size
VOCAB_SIZE = 1000              # Vocabulary size  
EMBED_DIM = 128                # Embedding dimension
DEVICE = "cuda"                # Device (cuda/cpu)
IMAGE_SIZE = 1024              # Poster size
NUM_BOXES = 10                 # Number of layout boxes
NUM_CLASSES = 5                # Number of element classes
```

## 📊 Supported Input Format

```python
{
    "cta": "Join Now",                                      # Call-to-action
    "target_audience": "Tech professionals",                # Audience description
    "product_description": "Annual Tech Conference 2026",   # Product/event description
    "style": "modern"                                       # Visual style
}
```

## 🎨 Pipeline Steps

1. **Content Processing**: Input is processed and optimized
2. **Layout Generation**: Layout model generates box positions and element types
3. **Background Generation**: Stable Diffusion creates a background based on product description
4. **Text Generation**: Intelligent text is generated for each element
5. **Rendering**: Text and elements are overlaid on the background
6. **Output**: Final poster is saved and metadata is returned

## ⚡ Performance Tips

1. **GPU**: Use CUDA for faster generation (10-20x faster than CPU)
2. **Batch Processing**: Generate multiple posters in sequence to reuse model
3. **Caching**: API server keeps models in memory between requests
4. **Resolution**: Adjust `IMAGE_SIZE` in config for speed vs quality tradeoff

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce image size in config.py
IMAGE_SIZE = 512  # Instead of 1024
```

### Models Not Found
Ensure model weights are in the correct paths:
- `layout_model.pth`
- Place in project root or modify paths in `PosterGenerator`

### Slow Generation
- First generation is slower (model loading)
- Use batch processing to reuse models
- Consider reducing `IMAGE_SIZE`

## 📦 Example Scripts

Run the examples:

```bash
# Simple usage example
python examples/example_1_simple_usage.py

# Batch generation example
python examples/example_2_batch_generation.py

# API client example (requires API server running)
python examples/example_3_api_client.py
```

## 🔌 Integration with Your Stack

### Django Integration
```python
from core.poster_generator import generate_poster_simple

def generate_poster_view(request):
    poster = generate_poster_simple(
        cta=request.POST.get("cta"),
        target_audience=request.POST.get("audience"),
        product_description=request.POST.get("description"),
        style=request.POST.get("style", "modern")
    )
    return FileResponse(poster, content_type="image/png")
```

### FastAPI Integration
Already provided in `api/server.py`

### Direct Library Usage
```python
from core.poster_generator import PosterGenerator

gen = PosterGenerator()
poster, metadata = gen.generate(
    cta="...", 
    target_audience="...", 
    product_description="...",
    style="..."
)
```

## 📝 License

Your license here

## 👥 Support

For issues and questions, please refer to the documentation or create an issue.

---

**Last Updated**: March 2026  
**Version**: 1.0.0
