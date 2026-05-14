"""
Fine-tune T5 model on service marketing copy generation task.

Input: service title + tone
Output: title | subtitle | description | cta
"""

import json
import os
import random

import torch
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


# Configuration
MODEL_NAME = "t5-base"
DATA_PATH = "data/service_marketing_train.json"
OUTPUT_DIR = "models/t5_service_marketing"

BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 256
RANDOM_SEED = 42
MAX_SAMPLES = None


def resolve_max_samples():
    """Resolve sample cap from env var for quick smoke tests."""
    raw = os.getenv("MAX_SAMPLES")
    if raw is None or raw.strip() == "":
        return MAX_SAMPLES
    value = int(raw)
    if value <= 0:
        raise ValueError("MAX_SAMPLES must be a positive integer when set")
    return value


def load_service_dataset(path: str):
    """Load and validate service-marketing dataset from JSON file."""
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    required_keys = {"title", "subtitle", "description", "cta", "tone"}
    cleaned = []
    for idx, row in enumerate(data):
        missing = required_keys - set(row.keys())
        if missing:
            raise ValueError(f"Row {idx} is missing required fields: {sorted(missing)}")
        cleaned.append(
            {
                "title": str(row["title"]).strip(),
                "subtitle": str(row["subtitle"]).strip(),
                "description": str(row["description"]).strip(),
                "cta": str(row["cta"]).strip(),
                "tone": str(row["tone"]).strip(),
            }
        )
    return cleaned


def build_training_pairs(rows):
    """Convert rows into T5 input/target pairs."""
    pairs = []
    for row in rows:
        input_text = (
            f"generate service marketing copy for: {row['title']} | tone: {row['tone']}"
        )
        target_text = (
            f"{row['title']} | {row['subtitle']} | {row['description']} | {row['cta']}"
        )
        pairs.append({"input": input_text, "target": target_text})
    return pairs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading dataset from {DATA_PATH}...")
    rows = load_service_dataset(DATA_PATH)
    print(f"Dataset size: {len(rows)}")

    pairs = build_training_pairs(rows)
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    max_samples = resolve_max_samples()
    if max_samples is not None:
        pairs = pairs[:max_samples]
        print(f"Debug mode active: using {len(pairs)} samples")

    train_size = int(0.9 * len(pairs))
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]

    print(f"Train size: {len(train_pairs)}")
    print(f"Val size: {len(val_pairs)}")

    train_dataset = Dataset.from_dict(
        {
            "input": [x["input"] for x in train_pairs],
            "target": [x["target"] for x in train_pairs],
        }
    )
    val_dataset = Dataset.from_dict(
        {
            "input": [x["input"] for x in val_pairs],
            "target": [x["target"] for x in val_pairs],
        }
    )

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    print(f"Model parameters: {model.num_parameters():,}")

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
        )

        labels = tokenizer(
            text_target=examples["target"],
            max_length=MAX_OUTPUT_LENGTH,
            truncation=True,
            padding="max_length",
        )

        # Ignore padding tokens in loss.
        label_ids = []
        for seq in labels["input_ids"]:
            label_ids.append(
                [token if token != tokenizer.pad_token_id else -100 for token in seq]
            )

        model_inputs["labels"] = label_ids
        return model_inputs

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        preprocess_function, batched=True, remove_columns=["input", "target"]
    )
    val_dataset = val_dataset.map(
        preprocess_function, batched=True, remove_columns=["input", "target"]
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=0.01,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    model_path = os.path.join(OUTPUT_DIR, "model")
    tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer")

    print(f"Saving model to {model_path}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)

    print("Training complete.")
    print(f"Saved model directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
