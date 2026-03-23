"""
Training script for custom text generation model.
Train on your own poster content data.

Usage:
    python train_text_generator.py --data_file your_training_data.csv --epochs 50 --batch_size 32

Data format (CSV):
    context,target
    Tech event description,TECH SUMMIT
    Annual technology conference,Annual Technology Summit
    ...
"""

import argparse
import csv
import torch
import logging
from torch.utils.data import DataLoader
from pathlib import Path

from models.custom_text_generator import (
    PosterTextGenerator, Vocab, TextGenerationDataset, TextGenerationTrainer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(csv_file: str) -> tuple:
    """
    Load training data from CSV file.
    CSV format: context,target
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        (contexts, targets) - Lists of input/output pairs
    """
    contexts = []
    targets = []
    
    logger.info(f"Loading training data from {csv_file}")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if 'context' in row and 'target' in row:
                context = row['context'].strip()
                target = row['target'].strip()
                
                if context and target:
                    contexts.append(context)
                    targets.append(target)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Loaded {i + 1} examples")
    
    logger.info(f"✅ Loaded {len(contexts)} training examples")
    return contexts, targets


def build_vocabulary(contexts: list, targets: list) -> Vocab:
    """
    Build vocabulary from training data.
    
    Args:
        contexts: List of input contexts
        targets: List of target texts
        
    Returns:
        Vocab object
    """
    logger.info("Building vocabulary...")
    vocab = Vocab()
    
    # Add words from contexts and targets
    for context in contexts + targets:
        for word in context.lower().split():
            vocab.add_word(word)
    
    logger.info(f"✅ Built vocabulary with {len(vocab.word2idx)} unique words")
    return vocab


def train_model(
    csv_file: str,
    output_dir: str = "models/trained",
    epochs: int = 50,
    batch_size: int = 32,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    n_layers: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train custom text generation model.
    
    Args:
        csv_file: Path to training data CSV
        output_dir: Directory to save model and vocabulary
        epochs: Number of training epochs
        batch_size: Batch size for training
        embed_dim: Embedding dimension
        hidden_dim: Hidden layer dimension
        n_layers: Number of LSTM layers
        device: Device to train on (cuda/cpu)
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    contexts, targets = load_training_data(csv_file)
    
    if len(contexts) == 0:
        logger.error("No training data loaded!")
        return
    
    # Build vocabulary
    vocab = build_vocabulary(contexts, targets)
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = TextGenerationDataset(contexts, targets, vocab, max_len=50)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    logger.info(f"Creating model on device: {device}")
    vocab_size = len(vocab.word2idx)
    model = PosterTextGenerator(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers
    )
    
    # Create trainer
    trainer = TextGenerationTrainer(model, vocab, device=device)
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Training examples: {len(contexts)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Device: {device}")
    
    for epoch in range(epochs):
        avg_loss = trainer.train_epoch(dataloader)
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    
    # Save model
    model_path = f"{output_dir}/poster_text_model.pth"
    vocab_path = f"{output_dir}/vocab.pkl"
    trainer.save(model_path, vocab_path)
    
    logger.info(f"✅ Training complete!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Vocabulary saved to: {vocab_path}")
    
    # Test the model
    logger.info("\nTesting model on first example...")
    test_context = contexts[0]
    test_target = targets[0]
    
    logger.info(f"Input: {test_context}")
    logger.info(f"Expected: {test_target}")
    
    input_ids = torch.tensor(vocab.encode(test_context, max_len=50), dtype=torch.long, device=device)
    generated = model.generate(input_ids, max_len=20, vocab=vocab)
    logger.info(f"Generated: {generated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom text generation model")
    parser.add_argument("--data_file", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--output_dir", type=str, default="models/trained", help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of LSTM layers")
    
    args = parser.parse_args()
    
    train_model(
        csv_file=args.data_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers
    )
