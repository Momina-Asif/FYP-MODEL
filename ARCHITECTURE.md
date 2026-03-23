# 🏗️ Architecture & System Design

Complete technical documentation of the Poster Generation Framework architecture.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ /api/        │  │ /api/        │  │ /api/        │       │
│  │ generate     │  │ generate-and │  │ styles       │       │
│  │              │  │ -metadata    │  │              │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│              Core Layer (PosterGenerator)                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  generate(cta, audience, description, style)      │    │
│  └────────────────────────────────────────────────────┘    │
└────────────┬────────────────────────────────────────────────┘
             │
     ┌───────┴────────┬──────────────┬──────────────┐
     │                │              │              │
┌────▼─────┐   ┌─────▼────┐  ┌─────▼────┐  ┌─────▼────┐
│ Layout   │   │ Content  │  │    SD    │  │ Renderer │
│ Model    │   │Generator │  │  Pipeline│  │          │
│          │   │          │  │          │  │          │
└──────────┘   └──────────┘  └──────────┘  └──────────┘
     │              │              │              │
     └──────────────┴──────────────┴──────────────┘
                    │
         ┌──────────▼──────────┐
         │  PIL Image Library  │
         │  (Rendering)        │
         └─────────────────────┘
```

## Component Architecture

### 1. API Layer (`api/server.py`)

**Responsibilities:**
- HTTP endpoint exposure
- Request validation
- Response formatting
- Error handling
- Health checks

**Key Endpoints:**
```
POST /api/generate                  → Binary image
POST /api/generate-and-metadata    → JSON with path + metadata
GET  /api/styles                    → Available styles
GET  /health                        → Service health
```

### 2. Core Layer (`core/poster_generator.py`)

**PosterGenerator Class:**

```python
class PosterGenerator:
    def generate(
        cta: str,
        target_audience: str,
        product_description: str,
        style: str
    ) -> Tuple[Image, Dict]
```

**Workflow:**
1. Initialize models (singleton on first call)
2. Generate layout using layout model
3. Create SD prompt from product description
4. Generate background using Stable Diffusion
5. Render text elements on background
6. Return image and metadata

### 3. Model Layer

#### 3.1 Layout Model (`models/layout_model.py`)
- Input: Random tensor (1, 40) - 10 boxes × 4 dimensions
- Output: Box coordinates (10, 4), Class logits (10, 5)
- Architecture: 2×512 FC layers + output projection

#### 3.2 Text-to-Layout Model (`models/text_to_layout.py`)
- Input: Tokenized text
- Output: Layout predictions
- Architecture: TextEncoder → Transformer → FC output

#### 3.3 Stable Diffusion (`models/sd_base.py`)
- Model: runwayml/stable-diffusion-v1-5
- Download: ~4GB on first use
- Uses: Diffusers library
- Optimization: xformers memory-efficient attention

### 4. Content Generation (`utils/content_generator_v2.py`)

**ContentGenerator Class:**

```python
class ContentGenerator:
    def generate_content(
        cta: str,
        product_description: str
    ) -> Dict[str, str]
```

**Content Types:**
- Title: Extracted keywords from product description
- Subtitle: Shortened product description
- Description: Relevant phrase based on keywords
- CTA: User-provided call-to-action

### 5. Rendering (`utils/renderer.py`)

**Process:**
1. Create PIL Image from background
2. Draw semi-transparent overlays for text areas
3. Render text with appropriate fonts
4. Apply positioning based on layout
5. Convert to PNG/JPG

## Data Flow

### Request Flow

```
HTTP Request
    ↓
FastAPI Route Handler
    ↓
PosterRequest Validation
    ↓
PosterGenerator.generate()
    ├→ _generate_layout()
    │   └→ LayoutModel forward pass
    ├→ _create_sd_prompt()
    ├→ _generate_background()
    │   └→ StableDiffusion pipeline
    ├→ _render_poster()
    │   ├→ ContentGenerator.generate_content()
    │   └→ PIL Image rendering
    ↓
Save to file (optional)
    ↓
HTTP Response (Image/JSON)
```

### Generation Pipeline

```
Input (CTA, Audience, Description)
    ↓
[Layout Generation]
    ├ Model input: Random (1, 40)
    └ Model output: Boxes (10, 4), Classes (10, 5)
    ↓
[Background Generation]
    ├ Create prompt: "{style} poster, {description}, professional"
    ├ Stable Diffusion inference: ~50 steps
    └ Output: 1024×1024 RGB image
    ↓
[Content Generation]
    ├ Extract keywords from description
    ├ Generate title, subtitle, description
    └ Use CTA as call-to-action text
    ↓
[Rendering]
    ├ Start with background image
    ├ For each layout element:
    │  ├ Draw semi-transparent overlay
    │  └ Render text with font
    └ Save final image
    ↓
Output (Image + Metadata)
```

## Memory Management

### Model Caching

```
First Call:
    Models not in memory
        ↓
    Load from disk (30-60 sec)
        ↓
    Store in memory
        ↓
    Process request

Subsequent Calls:
    Models in memory
        ↓
    Skip loading
        ↓
    Process request (~2 min)
```

### GPU Memory Usage

- **Stable Diffusion**: ~4-6 GB
- **Layout Model**: ~100 MB
- **Text Encoder**: ~500 MB
- **Total**: ~6-8 GB

## Configuration Hierarchy

```
config.py (defaults)
    ↓
Environment variables (overrides)
    ↓
.env file (local overrides)
    ↓
Runtime parameters (final overrides)
```

## Error Handling

### Graceful Degradation

```python
Layout Model not found
    → Use random initialization
    → Log warning
    → Continue

GPU not available
    → Fall back to CPU
    → Log warning
    → Continue (slower)

Font not available
    → Use default font
    → Log warning
    → Continue
```

### Error Categories

1. **Configuration Errors** (handled at startup)
2. **Model Loading Errors** (fallback to random init)
3. **GPU/CUDA Errors** (fallback to CPU)
4. **Image Processing Errors** (skip problematic element)
5. **File I/O Errors** (log and continue)

## Performance Characteristics

### Time Breakdown (RTX 3090)

```
First Generation:
  Model Loading:        30-60 sec
  Layout Generation:    0.1 sec
  Background (50 steps): 90 sec
  Rendering:            5 sec
  ─────────────────────────────
  Total:               125-155 sec (2-2.5 min)

Subsequent:
  Layout Generation:    0.1 sec
  Background (50 steps): 70 sec
  Rendering:            5 sec
  ─────────────────────────────
  Total:               75 sec (1-2 min)
```

### Memory Usage

```
Idle:           ~2 GB (models loaded)
During Gen:     ~8 GB (peak)
After Gen:      ~2 GB (models cached)
```

### Scalability

- **Batch**: Can process multiple posters sequentially
- **Parallel**: Each process needs its own GPU
- **CPU**: ~5x slower than GPU

## Extension Points

### Custom Layout Model

1. Replace `models/layout_model.py`
2. Update input/output shapes in config
3. PosterGenerator will use new model

### Custom SD Model

1. Modify `models/sd_base.py`:
   ```python
   pipe = StableDiffusionPipeline.from_pretrained(
       "YOUR_MODEL_ID"  # Change this
   )
   ```

### Custom Content Generator

1. Subclass `ContentGenerator`
2. Override `generate_content()`
3. Pass to `PosterGenerator.__init__()`

### Custom Renderer

1. Extend `utils/renderer.py`
2. Implement custom rendering logic
3. Update `_render_poster()` in `PosterGenerator`

## Deployment Architecture

### Single Server

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP
       ↓
┌─────────────────────┐
│  FastAPI Server     │
│  (1 process)        │
│  (poses limit: 1)   │
└─────────────────────┘
       ↓
   GPU/CPU
```

### Load Balanced

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP
       ↓
┌─────────────────────┐
│  Load Balancer      │
│  (NGINX/HAProxy)    │
└─────┬───────────────┘
      │
    ┌─┴──┬──┬─┐
    ↓    ↓  ↓ ↓
┌─────┐┌─────┐
│API-1││API-2│
└─────┘└─────┘
```

### Microservices

```
┌──────────────┐
│ API Gateway  │
└──────┬───────┘
       │
    ┌──┴────┬─────────┐
    ↓       ↓         ↓
┌────────┐┌────────┐┌──────────┐
│Layout  ││Content ││Background│
│Service ││Service ││Service   │
└────────┘└────────┘└──────────┘
```

## Security Considerations

### Input Validation
```python
- String length limits
- Special character filtering
- Type checking
- Rate limiting
```

### Output Security
```python
- No sensitive data in images
- Metadata sanitization
- File permission restrictions
```

### API Security
```python
- HTTPS only (in production)
- API key authentication
- CORS configuration
- Request rate limiting
```

## Monitoring & Observability

### Key Metrics

```
- Requests/minute
- Average generation time
- GPU utilization
- Memory usage
- Error rate
- Cache hit ratio
```

### Logging

```
INFO:   Request received
DEBUG:  Model loaded
DEBUG:  Layout generated
DEBUG:  Background generated
DEBUG:  Posting rendered
INFO:   Request completed
ERROR:  [Any errors]
```

## Database Schema (Optional)

```sql
CREATE TABLE posters (
    id VARCHAR(36) PRIMARY KEY,
    cta VARCHAR(255),
    target_audience TEXT,
    product_description TEXT,
    style VARCHAR(50),
    image_path VARCHAR(255),
    metadata JSON,
    created_at TIMESTAMP,
    user_id VARCHAR(36)
);

CREATE INDEX idx_created_at ON posters(created_at);
CREATE INDEX idx_user_id ON posters(user_id);
```

## Future Enhancements

1. **Multi-GPU Support**: Distribute models across GPUs
2. **Model Caching**: LRU cache with configurable size
3. **Custom Fonts**: Support user-provided fonts
4. **Advanced Layouts**: More sophisticated layout models
5. **Real-time Collaboration**: Multi-user editing
6. **Template System**: Pre-designed layouts
7. **A/B Testing**: Generate variations
8. **Analytics**: Track successful designs

---

For implementation details, see the source code and inline documentation.
