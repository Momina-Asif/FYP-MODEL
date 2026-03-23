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
from complete_pipeline_clean import generate_complete_poster

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
            "image": "output/image_xyz.png",
            "poster": "output/poster_xyz.png",
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
    print("🚀 POSTER GENERATION API - FLASK SERVER")
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
