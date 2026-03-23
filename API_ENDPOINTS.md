# API Endpoints for Poster & Social Media Ad Generation

## 🎯 Quick Answer: Which Endpoint to Use?

### ✅ For STRIKING Social Media Ads (NEW - RECOMMENDED)
**Endpoint**: `POST /api/generate-social-media-ad`

This uses the advanced prompt engineering to create visually striking ads with:
- Product-first prompt structure (product shown correctly)
- Dramatic lighting and professional composition
- Proper backgrounds (lifestyle, urban, nature, abstract)
- Perfect for Instagram, Facebook, social media

### ⚠️ For Basic Poster Generation (OLD)
**Endpoint**: `POST /api/generate`

Older endpoint - still works but less optimized for striking visuals

---

## 🚀 How to Use

### Start the Server
```bash
cd c:\Users\User\FYP-MODEL
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Then visit: `http://0.0.0.0:8000/docs` (Interactive API documentation)

---

## 📱 NEW ENDPOINT: Generate Striking Social Media Ads

### URL
```
POST http://0.0.0.0:8000/api/generate-social-media-ad
```

### Request Body
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

### Parameters

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| **service_or_product** | string | Any product name | What to advertise (e.g., "premium wireless headphones") |
| **ideal_market** | string | Any audience | Target audience (e.g., "tech professionals", "fitness enthusiasts") |
| **background_context** | string | `lifestyle`, `nature`, `urban`, `abstract`, `product-only` | Type of background |
| **lighting_style** | string | `dramatic`, `cinematic`, `natural`, `luxury` | Lighting approach |
| **mode** | string | `cinematic`, `social` | Detail level - `cinematic` = very detailed |
| **quality** | string | `fast` (30 steps), `high` (50 steps), `ultra` (75 steps) | Generation quality |
| **format** | string | `png`, `jpg` | Output image format |

### Response
Returns the generated PNG/JPG image with headers showing:
- `X-Prompt`: The generated prompt
- `X-Product`: Product name
- `X-Market`: Target market

### Example: Fitness Product
```json
{
    "service_or_product": "premium yoga mat",
    "ideal_market": "fitness enthusiasts and yogis",
    "background_context": "lifestyle",
    "lighting_style": "natural",
    "mode": "cinematic",
    "quality": "high",
    "format": "png"
}
```

### Example: Luxury Product
```json
{
    "service_or_product": "luxury Italian leather bag",
    "ideal_market": "luxury fashion consumers",
    "background_context": "abstract",
    "lighting_style": "luxury",
    "mode": "cinematic",
    "quality": "ultra",
    "format": "png"
}
```

### Example: Tech Product
```json
{
    "service_or_product": "smart home security camera",
    "ideal_market": "homeowners and tech-savvy users",
    "background_context": "urban",
    "lighting_style": "dramatic",
    "mode": "cinematic",
    "quality": "high",
    "format": "png"
}
```

---

## 🔍 DEBUG ENDPOINT: See Generated Prompts

### URL
```
POST http://0.0.0.0:8000/api/generate-social-media-ad-metadata
```

**Same parameters as above**, but returns JSON with:
- ✅ Generated prompt text
- ✅ Prompt length
- ✅ Negative prompt (artifact filtering)
- ✅ Generation parameters
- ✅ Your API input

### Response Example
```json
{
    "success": true,
    "message": "Prompt generated successfully",
    "metadata": {
        "prompt": "A visually striking advertisement showcasing premium wireless noise-canceling headphones...",
        "prompt_length": 450,
        "negative_prompt": "blurry, low quality, distorted...",
        "generation_params": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "height": 768,
            "width": 768
        }
    }
}
```

Use this to:
- See what prompt is being generated
- Understand why images look a certain way
- Debug and optimize parameters

---

## 🧪 Test with cURL

### Generate Striking Ad
```bash
curl -X POST "http://0.0.0.0:8000/api/generate-social-media-ad" \
  -H "Content-Type: application/json" \
  -d '{
    "service_or_product": "premium wireless headphones",
    "ideal_market": "tech enthusiasts",
    "background_context": "lifestyle",
    "lighting_style": "dramatic",
    "mode": "cinematic",
    "quality": "high",
    "format": "png"
  }' \
  -o generated_ad.png
```

### Debug Prompts
```bash
curl -X POST "http://0.0.0.0:8000/api/generate-social-media-ad-metadata" \
  -H "Content-Type: application/json" \
  -d '{
    "service_or_product": "luxury leather backpack",
    "ideal_market": "fashion-forward professionals",
    "background_context": "urban",
    "lighting_style": "luxury",
    "mode": "cinematic"
  }'
```

---

## 📊 Background Contexts Explained

| Context | Best For | Result |
|---------|----------|--------|
| **lifestyle** | Fashion, fitness, home goods | Product in real-world usage, aspirational |
| **nature** | Outdoor gear, eco products | Natural environment, organic elements |
| **urban** | Tech, fashion, luxury | City setting, modern architecture |
| **abstract** | Premium products, creative brands | Artistic patterns, dynamic design |
| **product-only** | Clean showcase | Simple white background (least striking) |

---

## 💡 Tips for Best Results

### 1. Be Specific
❌ Bad: `"phone"`
✅ Good: `"premium iPhone 15 Pro Max"`

### 2. Match Audience to Product
❌ Bad: yoga mat → "corporate executives"
✅ Good: yoga mat → "fitness enthusiasts and yogis"

### 3. Use Appropriate Background
❌ Bad: luxury watch → `product-only`
✅ Good: luxury watch → `abstract` or `urban`

### 4. Quality Settings
- Use `ultra` quality (75 steps) for portfolio/paid ads
- Use `high` quality (50 steps) for quick generation
- Use `fast` quality (30 steps) for testing

### 5. Create Variety
Generate with different combinations:
- Different `background_context` values
- Different `lighting_style` values
- Different `mode` options

---

## Example: Complete Workflow

1. **Debug first** - Use metadata endpoint to see prompts
```bash
POST /api/generate-social-media-ad-metadata
```

2. **Review the prompt** - Understand what's being generated

3. **Adjust parameters** - Change background, lighting, quality

4. **Generate final** - Use the main endpoint
```bash
POST /api/generate-social-media-ad
```

5. **Download image** - Use the generated image

---

## 🚀 Integration with Backend

### Django Integration

```python
import requests

def generate_ad(product, market, background='lifestyle'):
    """Generate striking social media ad"""
    
    response = requests.post(
        'http://127.0.0.1:8000/api/generate-social-media-ad',
        json={
            'service_or_product': product,
            'ideal_market': market,
            'background_context': background,
            'lighting_style': 'dramatic',
            'mode': 'cinematic',
            'quality': 'high'
        }
    )
    
    if response.status_code == 200:
        return response.content  # Image bytes
    else:
        return None
```

---

## 📝 Summary

| Feature | Old Endpoint | New Endpoint |
|---------|-------------|--------------|
| URL | `/api/generate` | `/api/generate-social-media-ad` |
| Prompt Engineering | Basic | Advanced (product-first, detailed) |
| Background Control | No | Yes (lifestyle, nature, urban, abstract) |
| Lighting Control | No | Yes (dramatic, cinematic, natural, luxury) |
| Best For | Simple posters | Striking social media ads |
| **Recommendation** | ❌ Don't use | ✅ **USE THIS!** |

---

## ✨ Key Improvements

The new endpoint uses structed API data and generates prompts like:

```
"A visually striking advertisement showcasing {product}
the {product} prominently displayed in the center, hero shot
created for {market}, attention-grabbing layout
{lighting} lighting with realistic reflections, cinematic rendering
modern design with high contrast colors, professional composition
suitable for Instagram, Facebook, and social media ads
{background_context} background with complementary elements
high-resolution, ultra-detailed, 8k quality"
```

**This produces**: Professional, striking social media ads that actually show the product correctly with beautiful backgrounds! 🎨
