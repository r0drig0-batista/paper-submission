"""
Transform Municipal Metadata Datasets for NER Training (Single Model)

This script converts preprocessed JSONL datasets into Hugging Face `Dataset` objects
ready for transformer-based token classification (e.g., BERT, XLM-RoBERTa).
It tokenizes each example, aligns BIO labels with subtokens, and saves per-municipality
datasets for train/validation/test splits.

"""

import json
import os
from transformers import BertTokenizerFast, XLMRobertaTokenizerFast
from datasets import Dataset


# -----------------------------------------------------------
# Tokenizer selection
# -----------------------------------------------------------
# You can switch between the following tokenizers as needed:
# 1. BERTimbau (Portuguese)
# 2. XLM-RoBERTa (multilingual)
# 3. Custom DAPT model (domain-adapted BERT)
# Uncomment one line below to choose.
# -----------------------------------------------------------
tokenizer = BertTokenizerFast.from_pretrained("neuralmind/bert-large-portuguese-cased")
# tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")
# tokenizer = BertTokenizerFast.from_pretrained("src/metadata_identification/Continuous_learning/models/bert_dapt")


# -----------------------------------------------------------
# Label mappings (BIO format)
# -----------------------------------------------------------
label2id = {
    "O": 0,
    "B-DATA": 1, "I-DATA": 2,
    "B-HORARIO-INICIO": 3, "I-HORARIO-INICIO": 4,
    "B-HORARIO-FIM": 5, "I-HORARIO-FIM": 6,
    "B-LOCAL": 7, "I-LOCAL": 8,
    "B-TIPO-REUNIAO-EXTRAORDINARIA": 9, "I-TIPO-REUNIAO-EXTRAORDINARIA": 10,
    "B-TIPO-REUNIAO-ORDINARIA": 11, "I-TIPO-REUNIAO-ORDINARIA": 12,
    "B-NUMERO-ATA": 13, "I-NUMERO-ATA": 14,
    # President participants
    "B-PARTICIPANTE-PRESIDENTE-PRESENTE": 15, "I-PARTICIPANTE-PRESIDENTE-PRESENTE": 16,
    "B-PARTICIPANTE-PRESIDENTE-AUSENTE": 17, "I-PARTICIPANTE-PRESIDENTE-AUSENTE": 18,
    "B-PARTICIPANTE-PRESIDENTE-SUBSTITUIDO": 19, "I-PARTICIPANTE-PRESIDENTE-SUBSTITUIDO": 20,
    # Councillor participants
    "B-PARTICIPANTE-VEREADOR-PRESENTE": 21, "I-PARTICIPANTE-VEREADOR-PRESENTE": 22,
    "B-PARTICIPANTE-VEREADOR-AUSENTE": 23, "I-PARTICIPANTE-VEREADOR-AUSENTE": 24,
    "B-PARTICIPANTE-VEREADOR-SUBSTITUIDO": 25, "I-PARTICIPANTE-VEREADOR-SUBSTITUIDO": 26
}

id2label = {v: k for k, v in label2id.items()}


# -----------------------------------------------------------
# Function: process and align tokenized examples
# -----------------------------------------------------------
def process_examples(raw_examples):
    """
    Tokenize and align BIO labels with wordpieces for transformer input.

    Args:
        raw_examples (list[dict]): raw JSONL examples with "tokens" and "tags"
    Returns:
        list[dict]: tokenized examples with input_ids, attention_mask, labels, ata_id
    """
    processed = []

    for ex in raw_examples:
        tokens = ex["tokens"]
        labels = ex["tags"]

        tokenized = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding=False,
            add_special_tokens=True
        )

        word_ids = tokenized.word_ids()  # map each subtoken to its word index
        aligned_labels = []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # CLS, SEP, or padding → ignore in loss
                aligned_labels.append(-100)
            elif word_idx != prev_word_idx:
                # first subtoken of the word → use its label
                aligned_labels.append(label2id.get(labels[word_idx], 0))
            else:
                # subsequent subtokens → ignore in loss
                aligned_labels.append(-100)
            prev_word_idx = word_idx

        processed.append({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": aligned_labels,
            "ata_id": ex["ata_id"]
        })

    return processed


# -----------------------------------------------------------
# Main processing loop for all municipalities
# -----------------------------------------------------------
municipalities = ["alandroal", "covilha", "campomaior", "guimaraes", "fundao", "porto"]
base_input = "ecir_submission/data/metadata_final"
base_output = "ecir_submission/folder"

os.makedirs(base_output, exist_ok=True)

for m in municipalities:
    print(f"\nProcessing municipality: {m}")

    # --- Load JSONL splits ---
    with open(f"{base_input}/train/{m}_train.jsonl", encoding="utf-8") as f:
        train_raw = [json.loads(line) for line in f]
    with open(f"{base_input}/test/{m}_test.jsonl", encoding="utf-8") as f:
        test_raw = [json.loads(line) for line in f]
    with open(f"{base_input}/val/{m}_val.jsonl", encoding="utf-8") as f:
        val_raw = [json.loads(line) for line in f]

    # --- Process all examples ---
    for split_name, raw_data in zip(["train", "test", "val"], [train_raw, test_raw, val_raw]):
        examples = process_examples(raw_data)
        for ex in examples:
            ex["municipio"] = m

        dataset = Dataset.from_list(examples)
        save_dir = os.path.join(base_output, f"municipio_{m}")
        os.makedirs(save_dir, exist_ok=True)
        dataset.save_to_disk(os.path.join(save_dir, split_name))

        print(f"  Saved {len(dataset)} {split_name} examples for {m}")


# -----------------------------------------------------------
# Save label mappings
# -----------------------------------------------------------
mappings_path = os.path.join(base_output, "label_mappings.json")
with open(mappings_path, "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

print("\n✅ Dataset transformation completed successfully!")
print(f"Mappings saved to: {mappings_path}")
