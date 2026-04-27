"""
Quick inference script for the fine-tuned service marketing T5 model.

Example:
python test_t5_service_marketing.py --service "Car Wash Booking Service" --tone "Convenient"
"""

import argparse
import os

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


MODEL_DIR = "models/t5_service_marketing/model"
TOKENIZER_DIR = "models/t5_service_marketing/tokenizer"


def parse_generated_text(text: str):
    parts = [p.strip() for p in text.split("|")]
    if len(parts) >= 4:
        return {
            "title": parts[0],
            "subtitle": parts[1],
            "description": parts[2],
            "cta": parts[3],
        }
    return {
        "raw_output": text.strip(),
    }


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned T5 service-marketing model")
    parser.add_argument("--service", required=True, help="Service name, e.g., Car Wash Booking Service")
    parser.add_argument("--tone", required=True, help="Tone, e.g., Convenient")
    parser.add_argument("--max_new_tokens", type=int, default=96, help="Maximum new tokens to generate")
    args = parser.parse_args()

    if not os.path.isdir(MODEL_DIR) or not os.path.isdir(TOKENIZER_DIR):
        raise FileNotFoundError(
            "Model/tokenizer not found. Train first or check paths: "
            f"{MODEL_DIR} and {TOKENIZER_DIR}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    prompt = f"generate service marketing copy for: {args.service} | tone: {args.tone}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    parsed = parse_generated_text(generated)

    print("\nPrompt:")
    print(prompt)
    print("\nGenerated:")
    if "raw_output" in parsed:
        print(parsed["raw_output"])
    else:
        print(f"Title: {parsed['title']}")
        print(f"Subtitle: {parsed['subtitle']}")
        print(f"Description: {parsed['description']}")
        print(f"CTA: {parsed['cta']}")


if __name__ == "__main__":
    main()
