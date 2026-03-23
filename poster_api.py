#!/usr/bin/env python
"""
REST API for Dynamic Poster Generation
Endpoint: POST /api/generate
Input: {"product": "product description"}
Output: Complete poster with text, image, and layout
"""

from flask import Flask, request, jsonify, send_file
import os
import json
from datetime import datetime
from complete_pipeline import generate_complete_poster

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Create output directory
os.makedirs('api_output', exist_ok=True)

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Poster API is running',
        'timestamp': datetime.now().isoformat()
    }), 200


# ============================================================================
# MAIN API ENDPOINT
# ============================================================================

@app.route('/api/generate', methods=['POST'])
def generate_poster():
    """
    🎯 Main Endpoint: Generate complete poster from product description
    
    Request:
        POST /api/generate
        Content-Type: application/json
        {
            "product": "bohemian style pants",
            "model": "base"  // optional: "base" or "small"
        }
    
    Response (200 OK):
        {
            "success": true,
            "text": {
                "title": "Boho Chic",
                "subtitle": "Elegance & Style",
                "description": "...",
                "cta": "Shop Now"
            },
            "image": "/api/output/image_xyz.png",
            "poster": "/api/output/poster_xyz.png",
            "timestamp": "2026-03-22T10:30:45.123456"
        }
    
    Error Response (400/500):
        {
            "success": false,
            "error": "Error description",
            "timestamp": "2026-03-22T10:30:45.123456"
        }
    """
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'product' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: "product"',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        product_description = data.get('product', '').strip()
        model_type = data.get('model', 'base')  # Default to T5-BASE
        
        if not product_description:
            return jsonify({
                'success': False,
                'error': 'Product description cannot be empty',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        if len(product_description) > 500:
            return jsonify({
                'success': False,
                'error': 'Product description too long (max 500 characters)',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        if model_type not in ['base', 'small']:
            return jsonify({
                'success': False,
                'error': 'Invalid model type. Use "base" or "small"',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        print(f"\n{'='*80}")
        print(f"🎯 API REQUEST: Generate poster for '{product_description}'")
        print(f"   Model: T5-{model_type.upper()}")
        print(f"{'='*80}\n")
        
        # Run complete pipeline
        use_base = (model_type == 'base')
        result = generate_complete_poster(product_description, use_base=use_base)
        
        # Prepare response
        response = {
            'success': True,
            'text': result['text'],
            'image': result['image'],
            'poster': result['poster'],
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ API Response generated successfully")
        print(f"   Text: {result['text']['title']}")
        print(f"   Image: {result['image']}")
        print(f"   Poster: {result['poster']}\n")
        
        return jsonify(response), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ API ERROR: {error_msg}\n")
        
        return jsonify({
            'success': False,
            'error': f'Pipeline failed: {error_msg}',
            'timestamp': datetime.now().isoformat()
        }), 500


# ============================================================================
# FILE SERVING (Optional: serve generated files)
# ============================================================================

@app.route('/api/output/<filename>', methods=['GET'])
def serve_output(filename):
    """Serve generated output files"""
    try:
        file_path = os.path.join('output', filename)
        
        # Security: prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'error': 'Invalid filename'}), 403
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(400)
def bad_request(e):
    return jsonify({
        'success': False,
        'error': 'Bad request',
        'details': str(e),
        'timestamp': datetime.now().isoformat()
    }), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'details': str(e),
        'timestamp': datetime.now().isoformat()
    }), 500


# ============================================================================
# INFO ENDPOINTS
# ============================================================================

@app.route('/api/info', methods=['GET'])
def api_info():
    """Get API information"""
    return jsonify({
        'name': 'Poster Generation API',
        'version': '1.0',
        'description': 'Dynamic AI poster generator with T5 text + Stable Diffusion images',
        'endpoints': {
            'GET /health': 'Health check',
            'GET /api/info': 'API information',
            'POST /api/generate': 'Generate poster from product description',
            'GET /api/output/<filename>': 'Serve generated files'
        },
        'models': {
            'text_generation': 'T5-BASE (223M params) or T5-SMALL (60M params)',
            'image_generation': 'Stable Diffusion 3 Medium (HF Router API)',
            'layout': 'PIL/Pillow'
        },
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/requirements', methods=['GET'])
def requirements():
    """Get environment requirements"""
    return jsonify({
        'environment_variables': {
            'HF_API_KEY': 'HuggingFace API token for Stable Diffusion image generation'
        },
        'python_packages': {
            'flask': 'Web framework',
            'torch': 'PyTorch for T5 model',
            'transformers': 'HuggingFace transformers',
            'pillow': 'Image processing',
            'requests': 'HTTP requests'
        },
        'setup': [
            '1. Set HF_API_KEY environment variable: export HF_API_KEY="your_token"',
            '2. Install dependencies: pip install -r requirements.txt',
            '3. Start API: python poster_api.py',
            '4. Test: curl -X POST http://localhost:5000/api/generate -H "Content-Type: application/json" -d \'{"product": "bohemian pants"}\''
        ]
    }), 200


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print(f"\n{'='*80}")
    print("🚀 POSTER GENERATION API")
    print(f"{'='*80}")
    print("\n📡 Starting Flask server...")
    print("   🌐 API URL: http://localhost:5000")
    print("   📝 Test endpoint: POST /api/generate")
    print("   🏥 Health check: GET /health")
    print("   ℹ️  API info: GET /api/info")
    print(f"\n{'='*80}\n")
    
    # Check HF_API_KEY
    if not os.getenv("HF_API_KEY"):
        print("⚠️  WARNING: HF_API_KEY not set!")
        print("   Image generation will fail without it.")
        print("   Set with: export HF_API_KEY='your_token'\n")
    else:
        print("✅ HF_API_KEY found\n")
    
    # Start server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Disable reloader to avoid loading models twice
    )

        """Load model once on startup"""
        print("🚀 Loading trained model...")
        
        config_path = 'models/trained/config.json'
        vocab_path = 'models/trained/poster_vocab.pkl'
        model_path = 'models/trained/structured_poster_model.pth'
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.input_vocab = vocab_data['input_vocab']
        self.output_vocab = vocab_data['output_vocab']
        self.input_word2idx = vocab_data['input_word2idx']
        self.output_word2idx = vocab_data['output_word2idx']
        self.idx2output_word = {i: w for i, w in enumerate(self.output_vocab)}
        
        # Initialize model
        class Encoder(nn.Module):
            def __init__(self, vocab_size, emb_dim, hid_dim):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=1)
                self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
            
            def forward(self, src):
                emb = self.embedding(src)
                _, (h, c) = self.lstm(emb)
                return h, c
        
        class Decoder(nn.Module):
            def __init__(self, vocab_size, emb_dim, hid_dim):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=1)
                self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
                self.fc = nn.Linear(hid_dim, vocab_size)
            
            def forward(self, trg, h, c):
                emb = self.embedding(trg)
                out, (h, c) = self.lstm(emb, (h, c))
                return self.fc(out), h, c
        
        class Seq2Seq(nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
            
            def forward(self, src):
                h, c = self.encoder(src)
                return h, c
        
        self.encoder = Encoder(len(self.input_vocab), self.config['EMB_DIM'], self.config['HID_DIM']).to(DEVICE)
        self.decoder = Decoder(len(self.output_vocab), self.config['EMB_DIM'], self.config['HID_DIM']).to(DEVICE)
        self.model = Seq2Seq(self.encoder, self.decoder).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        
        print("✅ Model loaded successfully")
    
    def generate_text(self, product_description):
        """Generate marketing text"""
        
        def tokenize(text):
            return text.lower().split()
        
        tokens = tokenize(product_description)
        tokens = [2] + [self.input_word2idx.get(t, 0) for t in tokens] + [3]
        tokens = (tokens + [1] * self.config['MAX_LEN'])[:self.config['MAX_LEN']]
        
        with torch.no_grad():
            src = torch.LongTensor([tokens]).to(DEVICE)
            h, c = self.model(src)
            
            words = []
            dec_input = torch.LongTensor([[2]]).to(DEVICE)
            
            for _ in range(self.config['MAX_LEN']):
                logits, h, c = self.decoder(dec_input, h, c)
                idx = logits.argmax(2).item()
                
                if idx == 3:  # <eos>
                    break
                if idx not in [0, 1, 2]:
                    word = self.idx2output_word.get(idx, '<unk>')
                    if word not in ['<unk>', '<pad>', '<sos>', '<eos>']:
                        words.append(word)
                
                dec_input = logits.argmax(2)
        
        text = ' '.join(words)
        parts = text.split('|')
        
        return {
            'title': parts[0].strip() if len(parts) > 0 else 'Premium Product',
            'subtitle': parts[1].strip() if len(parts) > 1 else 'Exceptional quality',
            'description': parts[2].strip() if len(parts) > 2 else 'Discover excellence',
            'cta': parts[3].strip() if len(parts) > 3 else 'Shop Now'
        }
    
    def create_poster(self, text_dict, output_path):
        """Create poster image"""
        
        poster = Image.new('RGB', (1200, 600), color=(245, 245, 250))
        draw = ImageDraw.Draw(poster)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 60)
            subtitle_font = ImageFont.truetype("arial.ttf", 32)
            body_font = ImageFont.truetype("arial.ttf", 22)
            cta_font = ImageFont.truetype("arial.ttf", 32)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
            cta_font = ImageFont.load_default()
        
        draw.rectangle([(0, 0), (1200, 100)], fill=(25, 118, 210))
        draw.text((50, 30), text_dict['title'].upper()[:40], fill='white', font=title_font)
        
        draw.text((50, 130), "✨ " + text_dict['subtitle'][:50], fill=(25, 118, 210), font=subtitle_font)
        
        desc = text_dict['description'][:100]
        y = 220
        for line in [desc[i:i+50] for i in range(0, len(desc), 50)]:
            draw.text((50, y), line, fill=(60, 60, 60), font=body_font)
            y += 40
        
        btn_w, btn_h = 250, 70
        draw.rectangle([(50, 470), (300, 540)], fill=(76, 175, 80))
        draw.text((80, 485), text_dict['cta'].upper(), fill='white', font=cta_font)
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        poster.save(output_path)
        
        return output_path

    """API endpoint for generating posters"""
    
    def post(self, request):
        """
        POST /api/generate-poster/
        
        Request body:
        {
            "product_description": "Premium wireless headphones with noise cancellation",
            "cta": "Shop Now"  (optional)
        }
        """
        
        try:
            product_description = request.data.get('product_description')
            
            if not product_description:
                return Response(
                    {'error': 'product_description is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get service
            service = PosterGeneratorService()
            
            # Generate text
            text_dict = service.generate_text(product_description)
            
            # Override CTA if provided
            if 'cta' in request.data:
                text_dict['cta'] = request.data['cta']
            
            # Create poster
            output_path = f'/tmp/poster_{hash(product_description)}.png'
            service.create_poster(text_dict, output_path)
            
            # Return image
            return FileResponse(open(output_path, 'rb'), content_type='image/png')
        
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ============================================================================
# ADD TO DJANGO urls.py
# ============================================================================

"""
from django.urls import path
from .views import PosterGeneratorView

urlpatterns = [
    ...
    path('api/generate-poster/', PosterGeneratorView.as_view(), name='generate-poster'),
    ...
]
"""

print("✅ API ready to integrate with Django!")
print("\nUsage:")
print("POST /api/generate-poster/")
print('Body: {"product_description": "...", "cta": "Shop Now"}')
print("Response: PNG image file")
