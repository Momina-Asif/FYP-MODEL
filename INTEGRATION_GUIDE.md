# 🔌 Integration Guide

Complete guide for integrating the Poster Generator Framework into your applications.

## Table of Contents

1. [Python API Integration](#python-api-integration)
2. [HTTP API Integration](#http-api-integration)
3. [Web Framework Integration](#web-framework-integration)
4. [Production Deployment](#production-deployment)
5. [Performance Optimization](#performance-optimization)

---

## Python API Integration

### Basic Integration

```python
from core.poster_generator import generate_poster_simple

def create_marketing_poster(product_info):
    """Simple integration example"""
    poster = generate_poster_simple(
        cta=product_info['call_to_action'],
        target_audience=product_info['audience'],
        product_description=product_info['description'],
        style=product_info.get('style', 'modern'),
        output_path=f"posters/{product_info['id']}.png"
    )
    return poster
```

### Advanced Integration with Metadata

```python
from core.poster_generator import PosterGenerator
import json

class PosterService:
    def __init__(self):
        self.generator = PosterGenerator()
    
    def create_poster(self, content_dict, metadata_file=None):
        """Create poster and save metadata"""
        poster, metadata = self.generator.generate(
            cta=content_dict['cta'],
            target_audience=content_dict['audience'],
            product_description=content_dict['description'],
            style=content_dict.get('style', 'modern'),
            output_path=f"posters/{content_dict['id']}.png"
        )
        
        # Save metadata
        if metadata_file:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return poster, metadata
```

### Batch Processing

```python
from core.poster_generator import PosterGenerator
import multiprocessing

def process_poster(item):
    """Process single poster"""
    generator = PosterGenerator()
    poster, metadata = generator.generate(
        cta=item['cta'],
        target_audience=item['audience'],
        product_description=item['description'],
        style=item.get('style', 'modern'),
        output_path=f"output/{item['id']}.png"
    )
    return item['id'], metadata

def batch_generate_posters(items, num_workers=4):
    """Generate multiple posters in parallel"""
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(process_poster, items)
    return dict(results)

# Usage
items = [
    {'id': '1', 'cta': 'Buy', 'audience': 'Fashionistas', 'description': 'Summer Sale'},
    {'id': '2', 'cta': 'Join', 'audience': 'Techies', 'description': 'Tech Summit'},
]
results = batch_generate_posters(items)
```

---

## HTTP API Integration

### Using Requests Library

```python
import requests
import json

class PosterAPI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def generate(self, cta, audience, description, style="modern"):
        """Generate poster via API"""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "cta": cta,
                "target_audience": audience,
                "product_description": description,
                "style": style
            },
            timeout=300
        )
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"API Error: {response.text}")
    
    def get_styles(self):
        """Get available styles"""
        response = requests.get(f"{self.base_url}/api/styles")
        return response.json()["styles"]

# Usage
api = PosterAPI()
poster_data = api.generate(
    cta="Join Now",
    audience="Tech professionals",
    description="Tech Conference 2026",
    style="modern"
)

with open("poster.png", "wb") as f:
    f.write(poster_data)
```

### cURL Examples

```bash
# Generate poster
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "cta": "Join Now",
    "target_audience": "Tech professionals",
    "product_description": "Tech Summit 2026",
    "style": "modern"
  }' \
  --output poster.png

# Get available styles
curl "http://localhost:8000/api/styles"

# Check health
curl "http://localhost:8000/health"
```

### JavaScript/Node.js

```javascript
async function generatePoster(content) {
  const response = await fetch('http://localhost:8000/api/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      cta: content.cta,
      target_audience: content.audience,
      product_description: content.description,
      style: content.style || 'modern'
    })
  });
  
  if (response.ok) {
    return await response.blob();
  } else {
    throw new Error(`API Error: ${response.statusText}`);
  }
}

// Usage
const content = {
  cta: "Buy Now",
  audience: "Shoppers",
  description: "Spring Sale 2026",
  style: "vibrant"
};

generatePoster(content).then(blob => {
  // Use the blob (e.g., display, download, upload)
});
```

---

## Web Framework Integration

### Django Integration

```python
# views.py
from django.http import FileResponse
from django.views import View
from core.poster_generator import PosterGenerator
import io

class GeneratePosterView(View):
    def __init__(self):
        self.generator = PosterGenerator()
    
    def post(self, request):
        """Handle POST request to generate poster"""
        poster, metadata = self.generator.generate(
            cta=request.POST.get('cta'),
            target_audience=request.POST.get('audience'),
            product_description=request.POST.get('description'),
            style=request.POST.get('style', 'modern')
        )
        
        # Convert PIL image to bytes
        buffer = io.BytesIO()
        poster.save(buffer, format='PNG')
        buffer.seek(0)
        
        return FileResponse(buffer, content_type='image/png')

# urls.py
from django.urls import path
from .views import GeneratePosterView

urlpatterns = [
    path('api/generate-poster/', GeneratePosterView.as_view(), name='generate_poster'),
]
```

### Flask Integration

```python
# app.py
from flask import Flask, request, send_file, jsonify
from core.poster_generator import PosterGenerator
import io

app = Flask(__name__)
generator = PosterGenerator()

@app.route('/api/generate-poster', methods=['POST'])
def generate_poster():
    """Generate poster endpoint"""
    data = request.json
    
    try:
        poster, metadata = generator.generate(
            cta=data.get('cta'),
            target_audience=data.get('target_audience'),
            product_description=data.get('product_description'),
            style=data.get('style', 'modern')
        )
        
        # Convert to bytes
        buffer = io.BytesIO()
        poster.save(buffer, format='PNG')
        buffer.seek(0)
        
        return send_file(buffer, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/styles', methods=['GET'])
def get_styles():
    """Get available styles"""
    styles = ['modern', 'minimalist', 'vibrant', 'elegant', 'bold', 'playful', 'professional', 'artistic']
    return jsonify({'styles': styles})

if __name__ == '__main__':
    app.run(debug=True)
```

### FastAPI Integration (Direct)

```python
# app.py
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from core.poster_generator import PosterGenerator
import io
import os

app = FastAPI()
generator = PosterGenerator()

class PosterRequest(BaseModel):
    cta: str
    target_audience: str
    product_description: str
    style: str = "modern"

@app.post("/generate-poster")
async def generate_poster(request: PosterRequest):
    """Generate poster endpoint"""
    poster, metadata = generator.generate(
        cta=request.cta,
        target_audience=request.target_audience,
        product_description=request.product_description,
        style=request.style
    )
    
    # Save to temporary file
    temp_path = f"/tmp/poster_{id(poster)}.png"
    poster.save(temp_path)
    
    return FileResponse(temp_path, media_type="image/png")
```

---

## Production Deployment

### Docker Deployment

```bash
# Build image
docker build -t poster-generator:latest .

# Run container
docker run -p 8000:8000 --gpus all poster-generator:latest

# With docker-compose
docker-compose up -d
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: poster-generator
spec:
  replicas: 2
  selector:
    matchLabels:
      app: poster-generator
  template:
    metadata:
      labels:
        app: poster-generator
    spec:
      containers:
      - name: poster-generator
        image: poster-generator:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
```

### NGINX Reverse Proxy

```nginx
upstream poster_api {
    server localhost:8000;
    server localhost:8001;
}

server {
    listen 80;
    server_name poster-generator.example.com;
    
    location /api/ {
        proxy_pass http://poster_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
    
    location /health {
        proxy_pass http://poster_api;
    }
}
```

---

## Performance Optimization

### Model Caching

```python
from core.poster_generator import PosterGenerator

# Singleton pattern for model reuse
_generator = None

def get_generator():
    global _generator
    if _generator is None:
        _generator = PosterGenerator()
    return _generator

# Usage
generator = get_generator()  # Reuses same instance
```

### Async Processing

```python
from fastapi import FastAPI, BackgroundTasks
import asyncio

app = FastAPI()

async def generate_poster_async(content, output_path):
    """Generate poster asynchronously"""
    generator = get_generator()
    poster, metadata = generator.generate(
        cta=content['cta'],
        target_audience=content['audience'],
        product_description=content['description'],
        style=content.get('style', 'modern'),
        output_path=output_path
    )
    return metadata

@app.post("/async-generate")
async def async_generate(content: dict, background_tasks: BackgroundTasks):
    """Queue poster generation in background"""
    task_id = str(time.time())
    output_path = f"output/{task_id}.png"
    
    background_tasks.add_task(generate_poster_async, content, output_path)
    
    return {"task_id": task_id, "status": "queued"}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Check generation status"""
    return {"task_id": task_id, "status": "processing"}
```

### Database Integration

```python
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.orm import declarative_base, Session
from datetime import datetime

Base = declarative_base()

class GeneratedPoster(Base):
    __tablename__ = "posters"
    
    id = Column(String, primary_key=True)
    cta = Column(String)
    audience = Column(String)
    description = Column(String)
    style = Column(String)
    image_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

def save_poster_record(db: Session, poster_data: dict, image_path: str):
    """Save poster generation record to database"""
    record = GeneratedPoster(
        id=poster_data['id'],
        cta=poster_data['cta'],
        audience=poster_data['audience'],
        description=poster_data['description'],
        style=poster_data['style'],
        image_path=image_path
    )
    db.add(record)
    db.commit()
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import FastAPI

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

@limiter.limit("5/minute")
@app.post("/api/generate")
auth generate_poster(request: PosterRequest):
    """Rate-limited endpoint"""
    pass
```

---

## Monitoring & Logging

### Structured Logging

```python
import logging
import json
from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

logger.info("Poster generated", extra={
    "generation_time": 2.5,
    "image_size": 1024,
    "elements": 10,
    "user_id": "user123"
})
```

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, start_http_server
import time

# Metrics
poster_generation_count = Counter('poster_generations_total', 'Total posters generated')
generation_time = Histogram('generation_time_seconds', 'Time to generate poster')
generation_errors = Counter('generation_errors_total', 'Generation errors')

@app.post("/api/generate")
async def generate_poster(request: PosterRequest):
    try:
        start_time = time.time()
        poster, metadata = generator.generate(...)
        duration = time.time() - start_time
        
        poster_generation_count.inc()
        generation_time.observe(duration)
        
        return FileResponse(...)
    except Exception as e:
        generation_errors.inc()
        raise

# Start metrics server
start_http_server(8001)
```

---

## Best Practices

1. **Singleton Pattern**: Use one generator instance for all requests
2. **Error Handling**: Implement comprehensive error handling
3. **Rate Limiting**: Limit API requests to prevent abuse
4. **Monitoring**: Track generation times and errors
5. **Caching**: Cache frequently requested posters
6. **Async**: Use async for non-blocking operations
7. **Documentation**: Document API endpoints clearly
8. **Testing**: Test with various inputs and edge cases

---

For more information, see [README_FRAMEWORK.md](README_FRAMEWORK.md) and [QUICKSTART.md](QUICKSTART.md)
