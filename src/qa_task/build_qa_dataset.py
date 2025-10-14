"""
SQuAD-style QA dataset builder for the minutes project.

For each document, this script generates two QA pairs:
  • Opening: the last sentence of the opening segment.
  • Closing: the first sentence of the closing segment.

It merges all municipalities, applies the predefined temporal split (train/val/test),
and saves the datasets in SQuAD v2 format.

Usage:
    python3 build_qa_dataset.py --lang pt
    python3 build_qa_dataset.py --lang en
"""

import json
import re
import argparse
from pathlib import Path
import spacy


# -------------------------------
# Helpers
# -------------------------------

def split_sentences(text: str):
    """
    Splits text into sentences using spaCy if available,
    otherwise falls back to a regex-based splitter.
    """
    if not text or not text.strip():
        return []
    try:
        nlp = spacy.load("pt_core_news_sm" if LANG == "pt" else "en_core_web_sm")
        doc = nlp(text.strip())
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        if sentences:
            return sentences
    except Exception:
        print("⚠️ spaCy failed, using regex fallback.")
    parts = re.split(r'(?<=[\.\?\!])[\s\n]+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_anchor(seg: dict, kind: str):
    """
    Extracts the anchor sentence (last for opening / first for closing)
    and computes its absolute start position in the full document.

    Args:
        seg (dict): Segment containing 'text', 'start', and 'end'.
        kind (str): Either 'opening' or 'closing'.

    Returns:
        tuple[str | None, int | None]: (answer_text, start_index)
    """
    if not seg or not seg.get("text"):
        return None, None

    # Get segment start position - try different possible key names
    seg_start = seg.get("start")
    if seg_start is None:
        seg_start = seg.get("begin")
    if seg_start is None:
        seg_start = seg.get("start_position")

    if seg_start is None:
        print(f"⚠️ Warning: No start position found in {kind} segment")
        return None, None

    # Clean the text before splitting into sentences
    seg_text = seg["text"].strip()

    # Remove trailing punctuation marks that aren't sentence-ending
    seg_text = re.sub(r'[-_\s]+$', '', seg_text)

    sentences = split_sentences(seg_text)
    if not sentences:
        print(f"⚠️ Warning: No sentences found in {kind} segment")
        return None, None

    if kind == "opening":
        chosen = sentences[-1]
        # Find the last occurrence of this sentence in the original segment text
        rel_start = seg["text"].rfind(chosen)
    else:
        chosen = sentences[0]
        # Find the first occurrence of this sentence in the original segment text
        rel_start = seg["text"].find(chosen)

    if rel_start == -1:
        print(f"⚠️ Warning: Could not find chosen sentence in {kind} segment text")
        return None, None

    start = seg_start + rel_start
    return chosen, start


def is_invalid_segment(seg_text: str) -> bool:
    """
    Checks if a segment is empty or invalid (e.g., only symbols or spaces).
    """
    if not seg_text or not seg_text.strip():
        return True
    txt = seg_text.strip()
    if re.fullmatch(r"[-_ \n\t]+", txt):
        return True
    if re.fullmatch(r"[\[\]\s\n\t]+", txt):
        return True
    if len(txt) < 5 and not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", txt):
        return True
    return False


def mk_qa_entry(municipality: str, doc_id: str, context: str, seg: dict | None, kind: str):
    """
    Builds a single QA entry (SQuAD v2 format) for either opening or closing.

    Args:
        municipality (str): Municipality name.
        doc_id (str): Document ID.
        context (str): Full text of the document.
        seg (dict): Segment dict with text and offsets.
        kind (str): 'opening' or 'closing'.

    Returns:
        dict: SQuAD v2-formatted QA entry.
    """
    if seg and is_invalid_segment(seg.get("text", "")):
        ans_text, ans_start = None, None
    else:
        ans_text, ans_start = extract_anchor(seg, kind)

    if ans_text and ans_start is not None and ans_text in context:
        answers = [{"text": ans_text, "answer_start": ans_start}]
        is_impossible = False
    else:
        answers = []
        is_impossible = True

    question = OPENING_Q if kind == "opening" else CLOSING_Q

    return {
        "title": municipality,
        "paragraphs": [{
            "context": context,
            "qas": [{
                "id": f"{doc_id}__{kind}",
                "question": question,
                "answers": answers,
                "is_impossible": is_impossible,
            }]
        }]
    }


def build_doc_items_for_municipality(docs: dict, municipality: str):
    """
    Builds QA pairs (opening + closing) for all documents of a municipality.

    Returns:
        list: List of two QA entries per document.
    """
    doc_groups = []
    for doc_id, doc in docs.items():
        ctx = doc.get("full_text", "")
        meta = doc.get("metadata", {}) or {}
        opening = meta.get("opening_segment")
        closing = meta.get("closing_segment")
        qa_open = mk_qa_entry(municipality, doc_id, ctx, opening, "opening")
        qa_close = mk_qa_entry(municipality, doc_id, ctx, closing, "closing")
        doc_groups.append([qa_open, qa_close])
    return doc_groups


def save_squad(items, out_path: Path):
    """Saves a list of QA entries in valid SQuAD v2 JSON format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"data": items, "version": "v2.0"}, f, ensure_ascii=False, indent=2)


def load_multiple(files):
    """
    Loads and merges multiple metadata JSON files.
    Returns a combined dict of all municipalities and their documents.
    """
    combined = {}
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = data.get("documents", {})
        for municipality, content in docs.items():
            if municipality not in combined:
                combined[municipality] = {}
            combined[municipality].update(content)
    return combined


# -------------------------------
# Main
# -------------------------------

def main():
    """
    Builds train/val/test SQuAD v2 datasets from metadata files.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", type=str, choices=["pt", "en"], default="pt",
                    help="Language: 'pt' for Portuguese or 'en' for English")
    ap.add_argument("--splits_json", type=Path,
                    default=Path("ecir_submission/split/splits.json"),
                    help="JSON file containing temporal split info (train/val/test)")
    args = ap.parse_args()

    global LANG, OPENING_Q, CLOSING_Q
    LANG = args.lang

    if args.lang == "pt":
        OPENING_Q = "No início da ata há um segmento de abertura. Qual é a última frase desse segmento de abertura?"
        CLOSING_Q = "No final da ata há um segmento de encerramento. Qual é a primeira frase desse segmento de encerramento?"
        input_jsons = [
            "ecir_submission/dataset_metadata_pt/alandroal_pt.json",
            "ecir_submission/dataset_metadata_pt/campomaior_pt.json",
            "ecir_submission/dataset_metadata_pt/covilha_pt.json",
            "ecir_submission/dataset_metadata_pt/fundao_pt.json",
            "ecir_submission/dataset_metadata_pt/guimaraes_pt.json",
            "ecir_submission/dataset_metadata_pt/porto_pt.json",
        ]
        output_dir = Path("ecir_submission/data/qa_squad")
    else:
        OPENING_Q = "At the beginning of the minutes there is an opening segment. What is the last sentence of that opening segment?"
        CLOSING_Q = "At the end of the minutes there is a closing segment. What is the first sentence of that closing segment?"
        input_jsons = [
            "ecir_submission/dataset_metadata_en/alandroal_en.json",
            "ecir_submission/dataset_metadata_en/campomaior_en.json",
            "ecir_submission/dataset_metadata_en/covilha_en.json",
            "ecir_submission/dataset_metadata_en/fundao_en.json",
            "ecir_submission/dataset_metadata_en/guimaraes_en.json",
            "ecir_submission/dataset_metadata_en/porto_en.json",
        ]
        output_dir = Path("ecir_submission/data/qa_squad_en")

    print(f"🔤 Selected language: {args.lang}")
    print(f"📂 Output directory: {output_dir}")

    # Load documents and splits
    minute_docs = load_multiple(input_jsons)

    with args.splits_json.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    train_files = set(splits["train_files"])
    val_files = set(splits["val_files"])
    test_files = set(splits["test_files"])

    train_all, val_all, test_all = [], [], []
    total_docs = 0

    # Assign documents to the correct split
    for municipality, docs in minute_docs.items():
        for doc_id, doc in docs.items():
            doc_groups = build_doc_items_for_municipality({doc_id: doc}, municipality)
            if f"{doc_id}.json" in train_files:
                train_all.extend(doc_groups[0])
            elif f"{doc_id}.json" in val_files:
                val_all.extend(doc_groups[0])
            elif f"{doc_id}.json" in test_files:
                test_all.extend(doc_groups[0])
            total_docs += 1

    # Save datasets
    save_squad(train_all, output_dir / "train.json")
    save_squad(val_all, output_dir / "val.json")
    save_squad(test_all, output_dir / "test.json")

    print(f"✅ Dataset successfully built ({args.lang})")
    print(f"📊 Total documents processed: {total_docs}")


if __name__ == "__main__":
    main()