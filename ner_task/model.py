"""
Train One Global NER Model for Municipal Minutes

This script trains a single transformer-based model (e.g., BERTimbau or XLM-RoBERTa)
on all municipalities combined, for the Named Entity Recognition (NER) task.
It loads tokenized datasets, fine-tunes the model, computes detailed metrics,
and saves model checkpoints, evaluation reports, and predictions.

Project: Metadata Extraction
"""

# -----------------------------------------------------------
# Imports
# -----------------------------------------------------------
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    BertTokenizerFast, BertForTokenClassification,
    XLMRobertaTokenizerFast, XLMRobertaForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from seqeval.metrics import classification_report, f1_score
import json
import numpy as np
import os


# -----------------------------------------------------------
# Metric containers
# -----------------------------------------------------------
f1_por_entidade_por_municipio = {}
precision_por_entidade_por_municipio = {}
recall_por_entidade_por_municipio = {}
macro_f1_por_municipio = {}

# -----------------------------------------------------------
# Load label mappings
# -----------------------------------------------------------
with open("ecir_submission/folder/label_mappings.json") as f:
    mappings = json.load(f)

label2id = mappings["label2id"]
id2label = {int(k): v for k, v in mappings["id2label"].items()}

folder_name = "teste"


# -----------------------------------------------------------
# Tokenizer and model selection
# -----------------------------------------------------------
# You can switch between the following options:
# 1. XLM-RoBERTa (multilingual)
# 2. BERTimbau (Portuguese)
# 3. Domain-adapted BERT (Continuous Learning)
# Uncomment the desired block.
# -----------------------------------------------------------

# --- Option 1: XLM-RoBERTa ---
"""
tokenizer_name = "xlm-roberta-large"
tokenizer = XLMRobertaTokenizerFast.from_pretrained(tokenizer_name)
model = XLMRobertaForTokenClassification.from_pretrained(
    tokenizer_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
"""

# --- Option 2: BERTimbau ---
tokenizer_name = "neuralmind/bert-large-portuguese-cased"
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
model = BertForTokenClassification.from_pretrained(
    tokenizer_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# --- Option 3: Continuous Learning BERT ---
"""
tokenizer = BertTokenizerFast.from_pretrained(
    "src/metadata_identification/Continuous_learning/models/bert_dapt"
)
model = AutoModelForTokenClassification.from_pretrained(
    "src/metadata_identification/Continuous_learning/models/bert_dapt",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
"""


# -----------------------------------------------------------
# Load datasets for all municipalities
# -----------------------------------------------------------
municipalities = ["alandroal", "covilha", "campomaior", "guimaraes", "fundao", "porto"]

datasets = {
    split: [load_from_disk(f"ecir_submission/folder/municipio_{m}/{split}")
            for m in municipalities]
    for split in ["train", "test", "val"]
}

train_dataset = concatenate_datasets(datasets["train"])
test_dataset = concatenate_datasets(datasets["test"])
val_dataset = concatenate_datasets(datasets["val"])


# -----------------------------------------------------------
# Compute evaluation metrics
# -----------------------------------------------------------
def compute_metrics(pred):
    """
    Compute token-level and entity-level F1 metrics using seqeval.
    Returns F1 for Trainer monitoring and stores detailed per-entity stats.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    true_labels, pred_labels = [], []

    for label_seq, pred_seq in zip(labels, preds):
        true_seq, pred_seq_clean = [], []
        for l, p in zip(label_seq, pred_seq):
            if l != -100:
                true_seq.append(id2label[l])
                pred_seq_clean.append(id2label[p])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_clean)

    f1 = f1_score(true_labels, pred_labels)
    report_dict = classification_report(true_labels, pred_labels, output_dict=True)

    # Extract detailed metrics for each entity
    entity_f1s = {k: v["f1-score"] for k, v in report_dict.items()
                  if k not in ["macro avg", "weighted avg", "micro avg"]}
    entity_precisions = {k: v["precision"] for k, v in report_dict.items()
                         if k not in ["macro avg", "weighted avg", "micro avg"]}
    entity_recalls = {k: v["recall"] for k, v in report_dict.items()
                      if k not in ["macro avg", "weighted avg", "micro avg"]}

    # Store for later use (post-training export)
    compute_metrics.last_entity_f1s = entity_f1s
    compute_metrics.last_entity_precisions = entity_precisions
    compute_metrics.last_entity_recalls = entity_recalls

    return {"f1": f1}


# -----------------------------------------------------------
# Training configuration
# -----------------------------------------------------------
args = TrainingArguments(
    output_dir=f"ecir_submission/ner_task/results/{folder_name}",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    fp16=True,
    num_train_epochs=15,
    weight_decay=0.01,
    logging_dir=f"ecir_submission/ner_task/results/{folder_name}",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=4,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# -----------------------------------------------------------
# Train the model
# -----------------------------------------------------------
trainer.train()

# -----------------------------------------------------------
# Save trained model and tokenizer
# -----------------------------------------------------------
save_path = f"ecir_submission/ner_task/results/{folder_name}"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)


# -----------------------------------------------------------
# Final evaluation on test set
# (kept fully intact as requested)
# -----------------------------------------------------------
eval_results = trainer.predict(test_dataset)
print(f"\nF1: {eval_results.metrics['test_f1']:.4f}")

micro_f1_global = eval_results.metrics['test_f1']
preds = eval_results.predictions.argmax(-1)
labels = eval_results.label_ids

true_labels, pred_labels = [], []

for label_seq, pred_seq in zip(labels, preds):
    true_seq, pred_seq_clean = [], []
    for l, p in zip(label_seq, pred_seq):
        if l != -100:
            true_seq.append(id2label[l])
            pred_seq_clean.append(id2label[p])
    true_labels.append(true_seq)
    pred_labels.append(pred_seq_clean)

report_global = classification_report(true_labels, pred_labels, output_dict=True)
micro_precision_global = report_global["micro avg"]["precision"]
micro_recall_global = report_global["micro avg"]["recall"]


# -----------------------------------------------------------
# Collect and save per-entity metrics
# -----------------------------------------------------------
f1s = compute_metrics.last_entity_f1s
precisions = compute_metrics.last_entity_precisions
recalls = compute_metrics.last_entity_recalls

f1_por_entidade_por_municipio["GLOBAL"] = f1s
precision_por_entidade_por_municipio["GLOBAL"] = precisions
recall_por_entidade_por_municipio["GLOBAL"] = recalls
macro_f1_por_municipio["GLOBAL"] = float(np.mean(list(f1s.values())))
micro_f1_por_municipio = {"GLOBAL": float(micro_f1_global)}
micro_precision_por_municipio = {"GLOBAL": float(micro_precision_global)}
micro_recall_por_municipio = {"GLOBAL": float(micro_recall_global)}

f1_medio_por_entidade = {ent: float(np.mean([val])) for ent, val in f1s.items()}


# -----------------------------------------------------------
# Save results and metrics
# -----------------------------------------------------------
base = f"ecir_submission/ner_task/results/{folder_name}"
os.makedirs(base, exist_ok=True)

metrics_to_save = {
    "f1_medio_por_entidade": f1_medio_por_entidade,
    "f1_por_entidade_por_municipio": f1_por_entidade_por_municipio,
    "precision_por_entidade_por_municipio": precision_por_entidade_por_municipio,
    "recall_por_entidade_por_municipio": recall_por_entidade_por_municipio,
    "macro_f1_por_municipio": macro_f1_por_municipio,
    "micro_f1_por_municipio": micro_f1_por_municipio,
    "micro_precision_por_municipio": micro_precision_por_municipio,
    "micro_recall_por_municipio": micro_recall_por_municipio,
}

for name, data in metrics_to_save.items():
    with open(f"{base}/{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------
# Save detailed token-level predictions
# -----------------------------------------------------------
predictions_path = f"{base}/predictions.jsonl"
with open(predictions_path, "w", encoding="utf-8") as f:
    for label_seq, pred_seq, input_ids in zip(labels, preds, test_dataset["input_ids"]):
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        tokens_clean, true_seq, pred_seq_clean = [], [], []

        for tok, l, p in zip(tokens, label_seq, pred_seq):
            if l != -100:  # skip special tokens
                tokens_clean.append(tok)
                true_seq.append(id2label[l])
                pred_seq_clean.append(id2label[p])

        f.write(json.dumps({
            "tokens": tokens_clean,
            "true": true_seq,
            "pred": pred_seq_clean
        }, ensure_ascii=False) + "\n")

print("\n✅ Training and evaluation completed successfully!")
print(f"Results saved to: {base}")
