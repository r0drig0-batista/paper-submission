import json
import os
from transformers import BertTokenizerFast, XLMRobertaTokenizerFast
from datasets import Dataset

# Load the tokenizer
# tokenizer = BertTokenizerFast.from_pretrained("neuralmind/bert-large-portuguese-cased")
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")


# Label mapping for BIO-format entity tags
label2id = {
    "O": 0,

    "B-DATA": 1, "I-DATA": 2,
    "B-HORARIO-INICIO": 3, "I-HORARIO-INICIO": 4,
    "B-HORARIO-FIM": 5, "I-HORARIO-FIM": 6,
    "B-LOCAL": 7, "I-LOCAL": 8,
    "B-TIPO-REUNIAO-EXTRAORDINARIA": 9, "I-TIPO-REUNIAO-EXTRAORDINARIA": 10,
    "B-TIPO-REUNIAO-ORDINARIA": 11, "I-TIPO-REUNIAO-ORDINARIA": 12,
    "B-NUMERO-ATA": 13, "I-NUMERO-ATA": 14,

    # PRESIDENTE
    "B-PARTICIPANTE-PRESIDENTE-PRESENTE": 15, "I-PARTICIPANTE-PRESIDENTE-PRESENTE": 16,
    "B-PARTICIPANTE-PRESIDENTE-AUSENTE": 17, "I-PARTICIPANTE-PRESIDENTE-AUSENTE": 18,
    "B-PARTICIPANTE-PRESIDENTE-SUBSTITUIDO": 19, "I-PARTICIPANTE-PRESIDENTE-SUBSTITUIDO": 20,

    # VEREADOR
    "B-PARTICIPANTE-VEREADOR-PRESENTE": 21, "I-PARTICIPANTE-VEREADOR-PRESENTE": 22,
    "B-PARTICIPANTE-VEREADOR-AUSENTE": 23, "I-PARTICIPANTE-VEREADOR-AUSENTE": 24,
    "B-PARTICIPANTE-VEREADOR-SUBSTITUIDO": 25, "I-PARTICIPANTE-VEREADOR-SUBSTITUIDO": 26
}

# Reverse mapping for decoding predictions later
id2label = {v: k for k, v in label2id.items()}


def process_examples(raw_examples):
    """
    Tokenize and align word-level BIO labels to subword tokens.
    Words split into multiple tokens will have only the first subtoken labeled,
    and the rest are ignored in the loss function (-100).
    """
    processed = []

    for ex in raw_examples:
        tokens = ex["tokens"]
        labels = ex["tags"]

        # Tokenize keeping the mapping between words and subwords
        tokenized = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding=False,
            add_special_tokens=True
        )

        word_ids = tokenized.word_ids()
        aligned_labels = []

        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # Ignore special tokens (CLS, SEP)
            elif word_idx != prev_word_idx:
                aligned_labels.append(label2id.get(labels[word_idx], 0))  # Label first subtoken
            else:
                aligned_labels.append(-100)  # Ignore subsequent subtokens
            prev_word_idx = word_idx

        processed.append({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": aligned_labels,
            "ata_id": ex["ata_id"]
        })

    return processed

# Input and output paths for cross-validation folds
base_input_dir = "ecir_submission/data/metadata_final"
base_output_dir = "ecir_submission/folder"
os.makedirs(base_output_dir, exist_ok=True)

# Iterate through each fold directory
for fold_dir in os.listdir(base_input_dir):
    fold_path = os.path.join(base_input_dir, fold_dir)
    if not os.path.isdir(fold_path):
        continue

    # Load raw JSONL data
    with open(os.path.join(fold_path, "train.jsonl"), encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]
    with open(os.path.join(fold_path, "test.jsonl"), encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]
    with open(os.path.join(fold_path, "val.jsonl"), encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]

    # Process all splits
    train_examples = process_examples(train_data)
    test_examples = process_examples(test_data)
    val_examples = process_examples(val_data)

    # Add fold name to track later
    for ex in train_examples + test_examples + val_examples:
        ex["fold"] = fold_dir

    # Convert lists into HuggingFace Datasets
    dataset_train = Dataset.from_list(train_examples)
    dataset_test = Dataset.from_list(test_examples)
    dataset_val = Dataset.from_list(val_examples)

    # Save processed datasets
    fold_output_dir = os.path.join(base_output_dir, fold_dir)
    os.makedirs(fold_output_dir, exist_ok=True)
    dataset_train.save_to_disk(os.path.join(fold_output_dir, "train"))
    dataset_val.save_to_disk(os.path.join(fold_output_dir, "val"))
    dataset_test.save_to_disk(os.path.join(fold_output_dir, "test"))

# Save label mappings for later use
with open(os.path.join(base_output_dir, "label_mappings.json"), "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f)


