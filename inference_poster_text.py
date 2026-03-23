#!/usr/bin/env python
"""
Inference script for structured poster text generation.
Loads trained model and generates title|subtitle|cta format.
"""

import torch
import torch.nn as nn
import pickle
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# LOAD CONFIGURATION AND VOCABULARIES
# ============================================================================

config_path = 'models/trained/config.json'
vocab_path = 'models/trained/poster_vocab.pkl'

print("📂 Loading model configuration and vocabularies...")

with open(config_path, 'r') as f:
    config = json.load(f)

with open(vocab_path, 'rb') as f:
    vocab_data = pickle.load(f)

input_vocab = vocab_data['input_vocab']
output_vocab = vocab_data['output_vocab']
input_word2idx = vocab_data['input_word2idx']
output_word2idx = vocab_data['output_word2idx']
idx2output_word = {i: w for i, w in enumerate(output_vocab)}

print(f"✅ Loaded vocabularies: input={len(input_vocab)}, output={len(output_vocab)}\n")

# ============================================================================
# LOAD MODEL
# ============================================================================

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
        logits = self.fc(out)
        return logits, h, c

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg=None):
        h, c = self.encoder(src)
        batch_sz = src.shape[0]
        max_len = 80
        vocab_sz = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_sz, max_len, vocab_sz, device=device)
        
        if trg is None:
            # Inference mode
            dec_input = torch.LongTensor([[2]] * batch_sz).to(device)  # <sos>
            
            for t in range(max_len):
                logits, h, c = self.decoder(dec_input, h, c)
                outputs[:, t] = logits.squeeze(1)
                dec_input = logits.argmax(2)
        else:
            # Training mode
            dec_input = trg[:, 0:1]
            for t in range(1, trg.shape[1]):
                logits, h, c = self.decoder(dec_input, h, c)
                outputs[:, t] = logits.squeeze(1)
                dec_input = trg[:, t:t+1]
        
        return outputs

print("🧠 Initializing model...")
encoder = Encoder(len(input_vocab), config['EMB_DIM'], config['HID_DIM']).to(device)
decoder = Decoder(len(output_vocab), config['EMB_DIM'], config['HID_DIM']).to(device)
model = Seq2Seq(encoder, decoder).to(device)

# Load trained weights
model_path = 'models/trained/structured_poster_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"✅ Model loaded from {model_path}\n")

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def generate_poster_text(product_description):
    """
    Generate structured poster text from product description.
    Returns: dict with title, subtitle, description, cta
    """
    
    # Tokenize input
    def tokenize(text):
        return text.lower().split()
    
    tokens = tokenize(product_description)
    tokens = [2] + [input_word2idx.get(t, 0) for t in tokens] + [3]  # Add <sos> <eos>
    tokens = (tokens + [1] * config['MAX_LEN'])[:config['MAX_LEN']]
    
    # Generate
    with torch.no_grad():
        src = torch.LongTensor([tokens]).to(device)
        outputs = model(src)
        preds = outputs.argmax(2)[0]
    
    # Decode
    words = []
    for idx in preds:
        idx = idx.item()
        if idx == 3:  # <eos>
            break
        if idx not in [0, 1, 2]:  # Skip <unk> <pad> <sos>
            word = idx2output_word.get(idx, '<unk>')
            if word not in ['<unk>', '<pad>', '<sos>', '<eos>']:
                words.append(word)
    
    text = ' '.join(words)
    
    # Parse structured output
    parts = text.split('|')
    
    result = {
        'title': parts[0].strip() if len(parts) > 0 else '',
        'subtitle': parts[1].strip() if len(parts) > 1 else '',
        'description': parts[2].strip() if len(parts) > 2 else '',
        'cta': parts[3].strip() if len(parts) > 3 else 'Shop Now'
    }
    
    return result

# ============================================================================
# TEST INFERENCE
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("TESTING MODEL INFERENCE")
    print("="*70 + "\n")
    
    test_products = [
        "Comfortable wireless headphones with noise cancellation and 30-hour battery life",
        "Premium leather wallet with RFID protection and minimalist design",
        "Stylish running shoes with cushioned sole and breathable mesh upper"
    ]
    
    for product in test_products:
        print(f"📦 Product: {product}")
        result = generate_poster_text(product)
        print(f"   🎯 Title: {result['title']}")
        print(f"   📝 Subtitle: {result['subtitle']}")
        print(f"   💬 CTA: {result['cta']}\n")
