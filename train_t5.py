"""
Fine-tune T5 model on marketing copy generation task
Input: Product context (e.g., "bohemian style")
Output: Title | Subtitle | Description | CTA
"""

import torch
import pandas as pd
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import os

# Configuration
MODEL_NAME = "t5-small"  # Use small model for faster training on CPU
BATCH_SIZE = 4
NUM_EPOCHS = 2
LEARNING_RATE = 5e-5
MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 256
OUTPUT_DIR = "models/t5_finetuned"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv("training_data/augmented_poster_training_data.csv")
print(f"Dataset size: {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Format data for T5
def format_data(batch):
    """Format data: input=context, target=title|subtitle|description|cta"""
    inputs = []
    targets = []
    
    for idx, row in batch.iterrows():
        # Input: "generate marketing for: <context>"
        input_text = f"generate marketing for: {row['context']}"
        
        # Target: "title | subtitle | description | cta"
        target_text = f"{row['title']} | {row['subtitle']} | {row['description']} | {row['cta']}"
        
        inputs.append(input_text)
        targets.append(target_text)
    
    return {"input": inputs, "target": targets}

# Prepare dataset
print("Preparing dataset...")
df_formatted = df.apply(lambda row: pd.Series({
    'input': f"generate marketing for: {row['context']}",
    'target': f"{row['title']} | {row['subtitle']} | {row['description']} | {row['cta']}"
}), axis=1)

# Split into train/validation
train_size = int(0.9 * len(df_formatted))
train_df = df_formatted[:train_size]
val_df = df_formatted[train_size:]

print(f"Train size: {len(train_df)}")
print(f"Val size: {len(val_df)}")

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_dict({
    "input": train_df['input'].tolist(),
    "target": train_df['target'].tolist()
})

val_dataset = Dataset.from_dict({
    "input": val_df['input'].tolist(),
    "target": val_df['target'].tolist()
})

# Load tokenizer and model
print(f"Loading {MODEL_NAME}...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

print(f"Model loaded. Total parameters: {model.num_parameters():,}")

# Tokenize function
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["target"]
    
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        targets,
        max_length=MAX_OUTPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess datasets
print("Tokenizing datasets...")
train_dataset = train_dataset.map(preprocess_function, batched=True, batch_size=32)
val_dataset = val_dataset.map(preprocess_function, batched=True, batch_size=32)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    logging_steps=50,
    save_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    gradient_accumulation_steps=1,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Train
print("Starting training...")
trainer.train()

# Save model
print(f"Saving model to {OUTPUT_DIR}...")
model.save_pretrained(f"{OUTPUT_DIR}/model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/tokenizer")

print("✅ Training complete!")
print(f"Model saved to {OUTPUT_DIR}")
