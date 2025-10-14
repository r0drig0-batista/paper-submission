import os
import json
import numpy as np
from datasets import load_from_disk
from transformers import (
    BertTokenizerFast, BertForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from seqeval.metrics import classification_report, f1_score

# === Metric containers ===
f1_per_entity_per_fold = {}
precision_per_entity_per_fold = {}
recall_per_entity_per_fold = {}
macro_f1_per_fold = {}

# === Load label mappings ===
with open("ecir_submission/folder/label_mappings.json") as f:
    mappings = json.load(f)

label2id = mappings["label2id"]
id2label = {int(k): v for k, v in mappings["id2label"].items()}

# === Tokenizer configuration ===
tokenizer_name = "neuralmind/bert-large-portuguese-cased"
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
# Alternative:
# tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")

base_dir = "ecir_submission/folder"
results_root = "ecir_submission/Resultados"
output_root = f"{results_root}/Output"
os.makedirs(output_root, exist_ok=True)

# Lists for global averages
all_micro_f1, all_micro_precision, all_micro_recall = [], [], []


def compute_metrics(pred):
    """Compute F1, precision, and recall (per-entity and overall)."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    true_labels, pred_labels = [], []
    for label_seq, pred_seq in zip(labels, preds):
        seq_true = [id2label[l] for l, p in zip(label_seq, pred_seq) if l != -100]
        seq_pred = [id2label[p] for l, p in zip(label_seq, pred_seq) if l != -100]
        true_labels.append(seq_true)
        pred_labels.append(seq_pred)

    f1 = f1_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, output_dict=True)

    # Store per-entity metrics (exclude global averages)
    entities = {k: v for k, v in report.items() if "avg" not in k}
    compute_metrics.last_entity_f1s = {k: v["f1-score"] for k, v in entities.items()}
    compute_metrics.last_entity_precisions = {k: v["precision"] for k, v in entities.items()}
    compute_metrics.last_entity_recalls = {k: v["recall"] for k, v in entities.items()}

    return {"f1": f1}


# === Iterate through folds ===
for fold_dir in sorted(os.listdir(base_dir)):
    fold_path = os.path.join(base_dir, fold_dir)
    if not os.path.isdir(fold_path):
        continue

    print(f"\n=== Training on fold: {fold_dir} ===")

    # Load datasets
    train_dataset = load_from_disk(os.path.join(fold_path, "train"))
    val_dataset = load_from_disk(os.path.join(fold_path, "val"))
    test_dataset = load_from_disk(os.path.join(fold_path, "test"))

    # Initialize model per fold
    model = BertForTokenClassification.from_pretrained(
        tokenizer_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # Training setup
    args = TrainingArguments(
        output_dir=f"{results_root}/{fold_dir}",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        fp16=True,
        num_train_epochs=15,
        weight_decay=0.01,
        logging_dir=f"{results_root}/{fold_dir}",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    # Save trained model
    model_path = f"{results_root}/models/{fold_dir}"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

    # === Evaluation on test set ===
    eval_results = trainer.predict(test_dataset)
    print(f"Fold {fold_dir} → Test F1: {eval_results.metrics['test_f1']:.4f}")

    preds = eval_results.predictions.argmax(-1)
    labels = eval_results.label_ids

    # Decode predictions and labels
    true_labels, pred_labels = [], []
    for label_seq, pred_seq in zip(labels, preds):
        seq_true = [id2label[l] for l, p in zip(label_seq, pred_seq) if l != -100]
        seq_pred = [id2label[p] for l, p in zip(label_seq, pred_seq) if l != -100]
        true_labels.append(seq_true)
        pred_labels.append(seq_pred)

    report = classification_report(true_labels, pred_labels, output_dict=True)
    micro_f1, micro_prec, micro_rec = (
        report["micro avg"]["f1-score"],
        report["micro avg"]["precision"],
        report["micro avg"]["recall"],
    )

    all_micro_f1.append(micro_f1)
    all_micro_precision.append(micro_prec)
    all_micro_recall.append(micro_rec)

    # Retrieve entity metrics
    f1s = compute_metrics.last_entity_f1s
    precisions = compute_metrics.last_entity_precisions
    recalls = compute_metrics.last_entity_recalls

    f1_per_entity_per_fold[fold_dir] = f1s
    precision_per_entity_per_fold[fold_dir] = precisions
    recall_per_entity_per_fold[fold_dir] = recalls
    macro_f1_per_fold[fold_dir] = float(np.mean(list(f1s.values())))

    # Save metrics for this fold
    fold_out = f"{results_root}/{fold_dir}"
    os.makedirs(fold_out, exist_ok=True)
    for name, data in {
        "f1_per_entity.json": f1s,
        "precision_per_entity.json": precisions,
        "recall_per_entity.json": recalls,
    }.items():
        with open(f"{fold_out}/{name}", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# === Aggregate global metrics across all folds ===
def avg_dicts(dicts):
    """Aggregate metrics by averaging per-entity across folds."""
    global_vals = {}
    for fold_vals in dicts.values():
        for ent, val in fold_vals.items():
            global_vals.setdefault(ent, []).append(val)
    return {ent: float(np.mean(vals)) for ent, vals in global_vals.items()}


global_f1 = avg_dicts(f1_per_entity_per_fold)
global_precision = avg_dicts(precision_per_entity_per_fold)
global_recall = avg_dicts(recall_per_entity_per_fold)

global_macro_f1 = float(np.mean(list(global_f1.values())))
global_micro_f1 = float(np.mean(all_micro_f1))
global_micro_precision = float(np.mean(all_micro_precision))
global_micro_recall = float(np.mean(all_micro_recall))

# Save aggregated metrics
global_metrics = {
    "f1_global.json": global_f1,
    "precision_global.json": global_precision,
    "recall_global.json": global_recall,
    "macro_f1_global.json": {"macro_f1": global_macro_f1},
    "micro_f1_global.json": {"micro_f1": global_micro_f1},
    "micro_precision_global.json": {"micro_precision": global_micro_precision},
    "micro_recall_global.json": {"micro_recall": global_micro_recall},
}

for name, data in global_metrics.items():
    with open(f"{output_root}/{name}", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print("\n=== Global results (averaged across folds) ===")
print(f"Macro-F1 global: {global_macro_f1:.4f}")
