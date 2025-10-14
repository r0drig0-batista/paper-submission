#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Question Answering (QA) training script (SQuAD v2 format) using BERTimbau-Large or XLM-RoBERTa.

Pipeline:
  1. Load train/validation/test in SQuAD v2 format ("data" → "paragraphs" → "qas")
  2. Flatten into individual examples (id, question, context, answers, is_impossible)
  3. Tokenize with sliding window (doc_stride)
  4. Train and evaluate (EM/F1) using HuggingFace `evaluate` ("squad_v2")
  5. Save checkpoints, metrics, and predictions

Example:
  python train_qa.py \
    --train_file data/qa_squad/train.json \
    --validation_file data/qa_squad/val.json \
    --test_file data/qa_squad/test.json \
    --model_name neuralmind/bert-large-portuguese-cased \
    --output_dir outputs/bertimbau-large-qa \
    --num_train_epochs 3 --per_device_train_batch_size 8
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
from datasets import Dataset, DatasetDict
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ----------------------------
# Load and flatten SQuAD v2
# ----------------------------
def read_squad_to_dataset(path: str) -> Dataset:
    """Reads a SQuAD v2-style file and returns a flat HuggingFace Dataset."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    for art in raw.get("data", []):
        for par in art.get("paragraphs", []):
            context = par.get("context", "")
            for qa in par.get("qas", []):
                examples.append({
                    "id": qa.get("id"),
                    "question": qa.get("question", ""),
                    "context": context,
                    "answers": qa.get("answers", []),
                    "is_impossible": qa.get("is_impossible", False),
                })
    return Dataset.from_list(examples)


# ----------------------------
# Post-processing (logits → text)
# ----------------------------
def postprocess_qa_predictions(
    examples,
    features,
    raw_predictions,
    tokenizer,
    n_best_size: int = 20,
    max_answer_length: int = 320,
    null_score_diff_threshold: float = 0.0,
):
    """
    Converts model logits into final text predictions per example ID.

    Notes:
      - Uses the lowest null score across windows (SQuAD v2 rule)
      - Uses the CLS token index for null predictions
      - Allows longer answers (up to `max_answer_length`)
    """
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}

    features_per_example = defaultdict(list)
    for i, f in enumerate(features):
        features_per_example[example_id_to_index[f["example_id"]]].append(i)

    predictions = {}
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []
        context = example["context"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            input_ids = features[feature_index]["input_ids"]

            cls_index = input_ids.index(tokenizer.cls_token_id)
            feature_null_score = float(start_logits[cls_index] + end_logits[cls_index])
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

            for s in start_indexes:
                for e in end_indexes:
                    if e < s:
                        continue
                    length = e - s + 1
                    if length > max_answer_length:
                        continue
                    if s >= len(offset_mapping) or e >= len(offset_mapping):
                        continue
                    if offset_mapping[s] is None or offset_mapping[e] is None:
                        continue

                    start_char = offset_mapping[s][0]
                    end_char = offset_mapping[e][1]
                    if start_char is None or end_char is None:
                        continue

                    text = context[start_char:end_char]
                    score = float(start_logits[s] + end_logits[e])
                    valid_answers.append({"text": text, "score": score})

        best_non_null = max(valid_answers, key=lambda x: x["score"]) if valid_answers else {"text": "", "score": -1e9}

        # SQuAD v2 null score rule
        if min_null_score is None:
            predictions[example["id"]] = best_non_null["text"]
        else:
            score_diff = min_null_score - best_non_null["score"]
            predictions[example["id"]] = "" if score_diff > null_score_diff_threshold else best_non_null["text"]

    return predictions

# ----------------------------
# Main training routine
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, choices=["pt", "en"], default="pt",
                        help="Language of the model: 'pt' for BERTimbau, 'en' for XLM-RoBERTa")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--n_best_size", type=int, default=50)
    parser.add_argument("--max_answer_length", type=int, default=120)
    parser.add_argument("--null_score_diff_threshold", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    # Model and path setup
    if args.lang == "pt":
        model_name = "deepset/xlm-roberta-large-squad2"
        base_path = "ecir_submission/data/qa_squad"
        args.output_dir = args.output_dir or "outputs/pt-roberta-large-qa"
    else:
        model_name = "deepset/xlm-roberta-large-squad2"
        base_path = "ecir_submission/data/qa_squad_en"
        args.output_dir = args.output_dir or "outputs/xlm-roberta-large-qa"

    args.train_file = f"{base_path}/train.json"
    args.validation_file = f"{base_path}/val.json"
    args.test_file = f"{base_path}/test.json"

    print(f"🔤 Language: {args.lang}")
    print(f"🧠 Model: {model_name}")
    print(f"📂 Dataset base path: {base_path}")

    # ----------------------------
    # Dataset loading
    # ----------------------------
    train_ds = read_squad_to_dataset(args.train_file)
    val_ds = read_squad_to_dataset(args.validation_file)
    raw_datasets = DatasetDict({"train": train_ds, "validation": val_ds})

    if os.path.exists(args.test_file):
        test_ds = read_squad_to_dataset(args.test_file)
        raw_datasets["test"] = test_ds

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    pad_on_right = tokenizer.padding_side == "right"

    # ----------------------------
    # Tokenization
    # ----------------------------
    def prepare_train_features(examples):
        """Tokenize QA pairs for training, aligning character to token positions."""
        questions = [q.strip() for q in examples["question"]]

        tokenized = tokenizer(
            questions if pad_on_right else examples["context"],
            examples["context"] if pad_on_right else questions,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=args.max_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions, end_positions = [], []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            is_impossible = examples.get("is_impossible", [False] * len(examples["id"]))[sample_index]

            # Handle impossible or missing answers
            if is_impossible or len(answers) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            answer = answers[0]
            start_char = answer["answer_start"]
            end_char = start_char + len(answer["text"])
            sequence_ids = tokenized.sequence_ids(i)
            idx = 1 if pad_on_right else 0

            # Locate context token span
            context_start, context_end = None, None
            for k, sid in enumerate(sequence_ids):
                if sid == idx and context_start is None:
                    context_start = k
                if sid == idx:
                    context_end = k

            if context_start is None or context_end is None:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            # Skip windows where answer is not fully inside the context
            if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            # Find token-level start/end indices
            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= start_char:
                token_start += 1
            token_start -= 1

            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= end_char:
                token_end -= 1
            token_end += 1

            start_positions.append(token_start)
            end_positions.append(token_end)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    def prepare_validation_features(examples):
        """Tokenize validation/test examples keeping offset mappings."""
        questions = [q.strip() for q in examples["question"]]
        tokenized = tokenizer(
            questions if pad_on_right else examples["context"],
            examples["context"] if pad_on_right else questions,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=args.max_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []

        for i in range(len(tokenized["input_ids"])):
            sample_index = sample_mapping[i]
            tokenized["example_id"].append(examples["id"][sample_index])

            sequence_ids = tokenized.sequence_ids(i)
            offsets = tokenized["offset_mapping"][i]
            idx = 1 if pad_on_right else 0
            tokenized["offset_mapping"][i] = [
                (o if sequence_ids[k] == idx else (None, None))
                for k, o in enumerate(offsets)
            ]

        return tokenized

    tokenized_train = raw_datasets["train"].map(
        prepare_train_features,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing training data",
    )
    tokenized_val = raw_datasets["validation"].map(
        prepare_validation_features,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        desc="Tokenizing validation data",
    )
    tokenized_test = (
        raw_datasets["test"].map(
            prepare_validation_features,
            batched=True,
            remove_columns=raw_datasets["test"].column_names,
            desc="Tokenizing test data",
        ) if "test" in raw_datasets else None
    )

    # ----------------------------
    # Evaluation and metrics
    # ----------------------------
    squad_metric = evaluate.load("squad_v2")

    def compute_metrics(p):
        predictions = postprocess_qa_predictions(
            raw_datasets["validation"],
            tokenized_val,
            p.predictions,
            tokenizer,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
        )

        references = []
        for ex in raw_datasets["validation"]:
            if ex["is_impossible"] or len(ex["answers"]) == 0:
                references.append({"id": ex["id"], "answers": {"text": [""], "answer_start": [0]}})
            else:
                texts = [a["text"] for a in ex["answers"]]
                starts = [a["answer_start"] for a in ex["answers"]]
                references.append({"id": ex["id"], "answers": {"text": texts, "answer_start": starts}})

        preds_list = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()]
        return squad_metric.compute(predictions=preds_list, references=references)

    # ----------------------------
    # Training
    # ----------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        save_total_limit=2,
        seed=args.seed,
        fp16=args.fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    trainer.save_model()
    with open(os.path.join(args.output_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save predictions
    val_preds = trainer.predict(tokenized_val).predictions
    val_text_preds = postprocess_qa_predictions(
        raw_datasets["validation"],
        tokenized_val,
        val_preds,
        tokenizer,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        null_score_diff_threshold=args.null_score_diff_threshold,
    )
    with open(os.path.join(args.output_dir, "val_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(val_text_preds, f, ensure_ascii=False, indent=2)

    # Optional: test predictions
    if tokenized_test is not None:
        test_preds = trainer.predict(tokenized_test).predictions
        test_text_preds = postprocess_qa_predictions(
            raw_datasets["test"],
            tokenized_test,
            test_preds,
            tokenizer,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
        )
        with open(os.path.join(args.output_dir, "test_predictions.json"), "w", encoding="utf-8") as f:
            json.dump(test_text_preds, f, ensure_ascii=False, indent=2)

        if "answers" in raw_datasets["test"].column_names:
            references = []
            for ex in raw_datasets["test"]:
                if ex["is_impossible"] or len(ex["answers"]) == 0:
                    references.append({"id": ex["id"], "answers": {"text": [""], "answer_start": [0]}})
                else:
                    texts = [a["text"] for a in ex["answers"]]
                    starts = [a["answer_start"] for a in ex["answers"]]
                    references.append({"id": ex["id"], "answers": {"text": texts, "answer_start": starts}})
            preds_list = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in test_text_preds.items()]
            test_metrics = squad_metric.compute(predictions=preds_list, references=references)
            with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, ensure_ascii=False, indent=2)
            print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
