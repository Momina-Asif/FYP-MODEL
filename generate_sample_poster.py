#!/usr/bin/env python
"""
End-to-end test: Generate poster with AI-generated text from inference model.
Shows full pipeline: product input → text generation → poster creation
"""

import torch
import torch.nn as nn
import pickle
import json
import os
import csv
import sys
from PIL import Image, ImageDraw, ImageFont

device = torch.device('cpu')

# ============================================================================
# LOAD MODEL & INFERENCE
# ============================================================================

print("📚 Loading inference model...\n")

config_path = 'models/trained/config.json'
vocab_path = 'models/trained/poster_vocab.pkl'

with open(config_path, 'r') as f:
    config = json.load(f)

with open(vocab_path, 'rb') as f:
    vocab_data = pickle.load(f)

input_vocab = vocab_data['input_vocab']
output_vocab = vocab_data['output_vocab']
input_word2idx = vocab_data['input_word2idx']
output_word2idx = vocab_data['output_word2idx']
idx2output_word = {i: w for i, w in enumerate(output_vocab)}

# Model architecture
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

encoder = Encoder(len(input_vocab), config['EMB_DIM'], config['HID_DIM']).to(device)
decoder = Decoder(len(output_vocab), config['EMB_DIM'], config['HID_DIM']).to(device)
model = Seq2Seq(encoder, decoder).to(device)

model.load_state_dict(torch.load('models/trained/structured_poster_model.pth', map_location=device))
model.eval()
print("✅ Model loaded\n")

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def tokenize(text):
    return text.lower().split()

def generate_poster_text(product_description):
    """Generate structured poster text"""
    
    tokens = tokenize(product_description)
    tokens = [2] + [input_word2idx.get(t, 0) for t in tokens] + [3]
    tokens = (tokens + [1] * config['MAX_LEN'])[:config['MAX_LEN']]
    
    with torch.no_grad():
        src = torch.LongTensor([tokens]).to(device)
        h, c = model(src)
        
        # Decode
        words = []
        dec_input = torch.LongTensor([[2]]).to(device)
        
        for _ in range(config['MAX_LEN']):
            logits, h, c = decoder(dec_input, h, c)
            idx = logits.argmax(2).item()
            
            if idx == 3:  # <eos>
                break
            if idx not in [0, 1, 2]:
                word = idx2output_word.get(idx, '<unk>')
                if word not in ['<unk>', '<pad>', '<sos>', '<eos>']:
                    words.append(word)
            
            dec_input = logits.argmax(2)
    
    text = ' '.join(words)
    parts = text.split('|')
    
    return {
        'title': parts[0].strip() if len(parts) > 0 else 'Premium Product',
        'subtitle': parts[1].strip() if len(parts) > 1 else '',
        'description': parts[2].strip() if len(parts) > 2 else '',
        'cta': parts[3].strip() if len(parts) > 3 else 'Shop Now'
    }

# ============================================================================
# CREATE POSTER
# ============================================================================

def create_poster_image(product_text_dict, output_path='output/generated_poster.png'):
    """Create poster image with generated text"""
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Create image
    width, height = 1200, 600
    img = Image.new('RGB', (width, height), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)
    
    # Try to use available fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 48)
        subtitle_font = ImageFont.truetype("arial.ttf", 32)
        body_font = ImageFont.truetype("arial.ttf", 24)
        cta_font = ImageFont.truetype("arial.ttf", 28)
    except:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        cta_font = ImageFont.load_default()
    
    # Layout: Left side colored, right side content
    # Draw header bar
    draw.rectangle([(0, 0), (width, 100)], fill=(25, 118, 210))
    
    # Draw title
    title_text = product_text_dict['title'][:40].upper()
    draw.text((50, 30), title_text, fill='white', font=title_font)
    
    # Draw subtitle
    subtitle_text = product_text_dict['subtitle'][:60]
    draw.text((50, 130), "✨ " + subtitle_text, fill=(25, 118, 210), font=subtitle_font)
    
    # Draw description (wrapped)
    desc_text = product_text_dict['description'][:100]
    y_pos = 220
    for line in [desc_text[i:i+50] for i in range(0, len(desc_text), 50)]:
        draw.text((50, y_pos), line, fill=(60, 60, 60), font=body_font)
        y_pos += 40
    
    # Draw CTA button
    cta_text = product_text_dict['cta'].upper()
    cta_x, cta_y = 50, 470
    button_width, button_height = 250, 70
    draw.rectangle(
        [(cta_x, cta_y), (cta_x + button_width, cta_y + button_height)],
        fill=(76, 175, 80)
    )
    draw.text(
        (cta_x + 35, cta_y + 18),
        cta_text,
        fill='white',
        font=cta_font
    )
    
    # Save
    img.save(output_path)
    return output_path

# ============================================================================
# MAIN TEST
# ============================================================================

print("="*70)
print("END-TO-END POSTER GENERATION TEST")
print("="*70 + "\n")

test_products = [
    "Premium wireless noise-canceling headphones with 30-hour battery and hi-fi sound quality",
    "Luxury Italian leather backpack with waterproof compartments and anti-theft design",
    "Smart fitness watch with heart rate monitoring, GPS, and 7-day battery life"
]

for i, product in enumerate(test_products, 1):
    print(f"📦 Test {i}: {product}\n")
    
    # Generate text
    print("  🤖 Generating poster text...")
    text_dict = generate_poster_text(product)
    print(f"     Title: {text_dict['title']}")
    print(f"     Subtitle: {text_dict['subtitle'][:50]}...")
    print(f"     CTA: {text_dict['cta']}\n")
    
    # Create poster
    print("  🎨 Creating poster image...")
    output_file = f'output/poster_example_{i}.png'
    create_poster_image(text_dict, output_file)
    print(f"     ✅ Saved: {output_file}\n")

print("="*70)
print("✅ COMPLETE!")
print("="*70)
print("\nGenerated posters in: output/poster_example_*.png")
print("\nNext steps:")
print("  1. Review generated posters")
print("  2. Train model longer for better text: python train_quick.py")
print("  3. Integrate with Stable Diffusion for product images")
