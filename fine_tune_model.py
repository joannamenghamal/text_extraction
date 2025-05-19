#!/usr/bin/env python3

import os
import argparse
import fitz  # PyMuPDF
import logging
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from huggingface_hub import login

DEFAULT_PDF = "CPSC_254_SYL.pdf"

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def chunk_text(text: str, block_size: int, stride: int):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + block_size]
        if len(chunk.strip()) > 10:
            chunks.append(chunk)
        start += stride
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TinyLLaMA on a PDF syllabus")
    parser.add_argument("--pdf_path", type=str, default=DEFAULT_PDF, help="Path to syllabus PDF")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output_dir", type=str, default="tiny_llama_cpsc254")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    # --- 1. Login to Hugging Face ---
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("❌ Please set the HUGGINGFACE_TOKEN environment variable.")
    login(token=hf_token)

    hf_username = os.getenv("HF_USERNAME", "your-username")

    # --- 2. Extract and Chunk PDF Text ---
    print("📄 Extracting text from PDF...")
    full_text = extract_text_from_pdf(args.pdf_path)
    print(f"✅ Extracted {len(full_text)} characters.")

    print("✂️ Chunking text...")
    texts = chunk_text(full_text, args.block_size, args.stride)
    print(f"✅ Created {len(texts)} text chunks.")

    # --- 3. Prepare Dataset ---
    ds = Dataset.from_dict({"text": texts})
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, padding_side="right")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.block_size)

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # --- 4. Data Collator ---
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # --- 5. Load Model ---
    print("🔄 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # --- 6. Set Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        logging_dir=f"{args.output_dir}/logs",
        push_to_hub=True,
        hub_model_id=f"{hf_username}/{args.output_dir}",
        hub_strategy="end",
        learning_rate=5e-5,
        weight_decay=0.01,
        no_cuda=args.no_cuda,
    )

    # --- 7. Start Training ---
    print("🚀 Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    try:
        trainer.train()
        trainer.push_to_hub(commit_message="Fine-tuned on CPSC 254 syllabus")
        print("✅ Model pushed to Hugging Face Hub successfully.")
    except Exception as e:
        print(f"❌ Training or upload failed: {e}")


if __name__ == "__main__":
    main()
