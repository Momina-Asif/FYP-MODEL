"""
Custom Text Generation Model for Poster Content
Simple Seq2Seq model that you can train on your own data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Tuple, List
import numpy as np
import pickle
import os

logger = logging.getLogger(__name__)


class Vocab:
    """Simple vocabulary manager"""
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_count = {}
    
    def add_word(self, word: str):
        """Add word to vocabulary"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_count[word] = self.word_count.get(word, 0) + 1
    
    def encode(self, text: str, max_len: int = 50) -> List[int]:
        """Convert text to indices"""
        tokens = text.lower().split()
        indices = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens[:max_len]]
        if len(indices) < max_len:
            indices += [self.word2idx["<PAD>"]] * (max_len - len(indices))
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Convert indices back to text"""
        words = [self.idx2word.get(idx, "<UNK>") for idx in indices if idx not in [0, 3]]  # Skip PAD and END
        return " ".join(words).strip()
    
    def save(self, path: str):
        """Save vocabulary"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_count': self.word_count
            }, f)
    
    def load(self, path: str):
        """Load vocabulary"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.word_count = data['word_count']


class Encoder(nn.Module):
    """Encoder - Processes input context"""
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, bidirectional=False)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len)
        Returns:
            output: (batch_size, seq_len, hidden_dim)
            (h, c): LSTM hidden states (n_layers, batch, hidden_dim)
        """
        embedded = self.embedding(x)
        output, (h, c) = self.lstm(embedded)
        return output, (h, c)


class Decoder(nn.Module):
    """Decoder - Generates output text"""
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 1)
            h, c: Hidden states from encoder
        Returns:
            logits: (batch_size, vocab_size)
        """
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, (h, c) = self.lstm(embedded, (h, c))
        logits = self.fc(output.squeeze(1))
        return logits, (h, c)


class PosterTextGenerator(nn.Module):
    """Seq2Seq model for poster text generation"""
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, n_layers)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, n_layers)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, input_seq_len)
            targets: (batch_size, target_seq_len)
            teacher_forcing_ratio: Probability of using teacher forcing
        Returns:
            outputs: (batch_size, target_seq_len, vocab_size)
        """
        batch_size, target_len = targets.shape
        outputs = torch.zeros(batch_size, target_len, self.vocab_size)
        
        # Encode input
        _, (h, c) = self.encoder(inputs)
        
        # Prepare initial decoder input (START token)
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long) * 2  # START token
        if inputs.device.type == 'cuda':
            decoder_input = decoder_input.cuda()
            outputs = outputs.cuda()
        
        # Decode
        for t in range(target_len):
            logits, (h, c) = self.decoder(decoder_input, h, c)
            outputs[:, t, :] = logits
            
            # Teacher forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = targets[:, t].unsqueeze(1)
            else:
                decoder_input = torch.argmax(logits, dim=1).unsqueeze(1)
        
        return outputs
    
    def generate(self, input_ids: torch.Tensor, max_len: int = 30, vocab: Vocab = None) -> str:
        """Generate text from input"""
        self.eval()
        with torch.no_grad():
            _, (h, c) = self.encoder(input_ids.unsqueeze(0))
            
            decoder_input = torch.ones(1, 1, dtype=torch.long) * 2  # START token
            if input_ids.device.type == 'cuda':
                decoder_input = decoder_input.cuda()
            
            generated_ids = []
            
            for _ in range(max_len):
                logits, (h, c) = self.decoder(decoder_input, h, c)
                token_id = torch.argmax(logits, dim=1)
                generated_ids.append(token_id.item())
                
                if token_id.item() == 3:  # END token
                    break
                
                decoder_input = token_id.unsqueeze(1)
        
        if vocab:
            return vocab.decode(generated_ids)
        return " ".join(str(idx) for idx in generated_ids)


class TextGenerationDataset(Dataset):
    """Dataset for training text generation model"""
    def __init__(self, contexts: List[str], targets: List[str], vocab: Vocab, max_len: int = 50):
        """
        Args:
            contexts: List of input contexts (product descriptions)
            targets: List of target texts (titles, subtitles, etc.)
            vocab: Vocabulary object
            max_len: Maximum sequence length
        """
        self.contexts = [vocab.encode(ctx, max_len) for ctx in contexts]
        self.targets = [vocab.encode(tgt, max_len) for tgt in targets]
        assert len(self.contexts) == len(self.targets)
    
    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.contexts[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


class TextGenerationTrainer:
    """Trainer for the text generation model"""
    def __init__(self, model: PosterTextGenerator, vocab: Vocab, device: str = 'cuda'):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD token
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs, targets, teacher_forcing_ratio=0.5)
            
            # Calculate loss
            loss = self.loss_fn(
                outputs.view(-1, self.model.vocab_size),
                targets.view(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save(self, model_path: str, vocab_path: str):
        """Save model and vocabulary"""
        torch.save(self.model.state_dict(), model_path)
        self.vocab.save(vocab_path)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vocab saved to {vocab_path}")
