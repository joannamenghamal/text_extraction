#!/usr/bin/env python3
# tiny_llama_ocr.py

"""
Uses a fine-tuned Tiny LLaMA model to extract purchased items and total cost
from receipt text stored in a CSV file (e.g., from OCR via Tesseract).
"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# --- CONFIGURATION ---
csv_path = "extracted_text.csv"             # Update this to your actual file
model_path = "./fine_tuned_tiny_llama"    # Update this to your model path

# --- Load Model ---
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)


# --- Read and Prepare Text from CSV ---
def read_csv_as_text(path):
    try:
        df = pd.read_csv(path, header=None)
        text = "\n".join(df[0].astype(str).tolist())  # assumes 1 column of text
        return text
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return ""

# --- Query LLaMA ---
def query_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Extract Items + Total ---
def extract_items_and_total(response_text):
    items = []
    total = None
    for line in response_text.strip().splitlines():
        if line.lower().startswith("item:"):
            try:
                parts = line.split(",")
                name = parts[0].split(":")[1].strip()
                price = float(parts[1].split(":")[1].strip())
                items.append((name, price))
            except (IndexError, ValueError):
                continue
        elif line.lower().startswith("total:"):
            try:
                total = float(line.split(":")[1].strip())
            except ValueError:
                continue
    return items, total

# --- MAIN ---
if __name__ == "__main__":
    print("=== Tiny LLaMA OCR CSV Analyzer ===\n")

    ocr_text = read_csv_as_text(csv_path)
    if not ocr_text:
        exit(1)

    print("📄 Loaded OCR Text from CSV:\n", ocr_text[:300], "...\n")  # preview first 300 chars

    prompt = f"""Extract the list of purchased items and their prices from this receipt text, and calculate the total:

{ocr_text}

Return the output in this format:
Item: <name>, Price: <amount>
...
Total: <amount>
"""

    print("🤖 Querying fine-tuned LLaMA model...\n")
    model_response = query_model(prompt)
    print("📥 Model Response:\n", model_response)

    items, total = extract_items_and_total(model_response)

    print("\n🧾 Parsed Items:")
    for item, price in items:
        print(f" - {item}: ${price:.2f}")
    if total is not None:
        print(f"\n💵 Parsed Total: ${total:.2f}")
    else:
        print("\n❗ Could not extract total.")
