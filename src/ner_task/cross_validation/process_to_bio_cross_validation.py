#!/usr/bin/env python
"""
Process Metadata Annotations - New Format

This script converts structured metadata from the new dataset format
into JSONL files used for token classification training.
It extracts entities (date, time, location, participants, etc.)
from the opening and closing segments of each meeting minute.
"""

import os
import json
import logging
import argparse
import re
from pathlib import Path
import spacy
from tqdm import tqdm
from document_chunker import DocumentChunker
from faker import Faker
import dateparser
import random
from datetime import datetime, timedelta

# === Logging configuration ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Load spaCy model (English version by default) ===
# To switch to Portuguese: model = "pt_core_news_sm"
model = "en_core_web_sm"

try:
    nlp = spacy.load(model)
    logger.info(f"Loaded spaCy model successfully.")
except Exception:
    logger.warning(f"Could not load spaCy model. Attempting installation...")
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", model])
    nlp = spacy.load(model)
    logger.info(f"Installed and loaded spaCy model.")

# === Custom tokenizer for date and time expressions ===
def custom_tokenize_datetime(text):
    """
    Tokenize text while correctly splitting date and time expressions.
    e.g. "03/12/2024 10:30" → ["03", "/", "12", "/", "2024", "10", ":", "30"]

    Returns:
        tokens (list of str): list of token texts
        token_positions (list of int): start positions of each token in the text
    """
    pattern = r'(\d{1,2}/\d{1,2}/\d{4})|(\d{1,2}[:.]\d{2})'
    matches = list(re.finditer(pattern, text))

    tokens, token_positions = [], []
    last_end = 0

    for match in matches:
        start, end = match.span()

        # Process normal text before a date/time match
        if start > last_end:
            pre_text = text[last_end:start]
            pre_doc = nlp(pre_text)
            for token in pre_doc:
                tokens.append(token.text)
                token_positions.append(last_end + token.idx)

        matched_text = match.group()

        # Split date expressions like "03/12/2024"
        if '/' in matched_text:
            day, month, year = matched_text.split('/')
            tokens.extend([day, '/', month, '/', year])
            token_positions.extend([
                start,
                start + len(day),
                start + len(day) + 1,
                start + len(day) + 1 + len(month),
                start + len(day) + 1 + len(month) + 1
            ])
        # Split time expressions like "10:30" or "10.30"
        elif ':' in matched_text or '.' in matched_text:
            sep = ':' if ':' in matched_text else '.'
            hour, minute = matched_text.split(sep)
            tokens.extend([hour, sep, minute])
            token_positions.extend([
                start,
                start + len(hour),
                start + len(hour) + 1
            ])

        last_end = end

    # Process remaining text after the last match
    if last_end < len(text):
        post_text = text[last_end:]
        post_doc = nlp(post_text)
        for token in post_doc:
            tokens.append(token.text)
            token_positions.append(last_end + token.idx)

    return tokens, token_positions


def find_end_time_in_closing(closing_text, metadata_end_time_text):
    """
    Try to locate the meeting end time inside the 'closing' segment text.

    Args:
        closing_text (str): text of the closing segment
        metadata_end_time_text (str): the time expression extracted from metadata

    Returns:
        dict or None: {"text": ..., "begin": ..., "end": ...} if found, otherwise None
    """
    if not closing_text or not metadata_end_time_text:
        return None

    # Try an exact string match (case-insensitive)
    idx = closing_text.lower().find(metadata_end_time_text.lower())
    if idx != -1:
        return {
            "text": metadata_end_time_text,
            "begin": idx,
            "end": idx + len(metadata_end_time_text)
        }

    # Fallback: search for typical time patterns in Portuguese/English
    time_patterns = [
        r'\b\d{1,2}[:.]\d{2}\s*horas?\b',             # e.g., "10:30 horas"
        r'\b\d{1,2}\s*horas?\s*e\s*\d{1,2}\s*minutos?\b'  # e.g., "10 horas e 30 minutos"
    ]

    for pattern in time_patterns:
        matches = list(re.finditer(pattern, closing_text, re.IGNORECASE))
        if matches:
            # Return the *last* match (more likely to represent the meeting end)
            match = matches[-1]
            return {
                "text": match.group(),
                "begin": match.start(),
                "end": match.end()
            }

    return None


def extract_entities_from_metadata(metadata, closing_text=""):
    """
    Extract structured entities (date, time, location, participants, etc.)
    from the metadata dictionary of a meeting document.

    Args:
        metadata (dict): structured metadata for a document
        closing_text (str): text of the closing segment (used for time correction)

    Returns:
        list of dict: each entity has fields like:
            {
                "type": "DATA",
                "begin": 123,
                "end": 131,
                "text": "03/12/2024",
                "metadata_type": "Data",
                "attributes": {...}
            }
    """
    entities = []

    # === Date ===
    if metadata.get("date"):
        date_info = metadata["date"]
        entities.append({
            "type": "DATA",
            "begin": date_info["begin"],
            "end": date_info["end"],
            "text": date_info["text"],
            "metadata_type": "Data",
            "attributes": {}
        })

    # === Start time ===
    if metadata.get("begin_time"):
        time_info = metadata["begin_time"]
        entities.append({
            "type": "HORARIO-INICIO",
            "begin": time_info["begin"],
            "end": time_info["end"],
            "text": time_info["text"],
            "metadata_type": "Horário",
            "attributes": {"Horrio": "início"}
        })

    # === End time ===
    if metadata.get("end_time"):
        end_time_info = metadata["end_time"]
        muni = metadata.get("municipality", "").lower()

        # Some municipalities store the end time only inside the closing segment
        closing_munis = ["alandroal", "campomaior", "guimaraes"]

        if any(m in muni for m in closing_munis) and closing_text:
            # Try to find and correct offsets within the closing section
            corrected_time = find_end_time_in_closing(closing_text, end_time_info["text"])
            if corrected_time:
                closing_segment = metadata.get("closing_segment", {})
                closing_offset = closing_segment.get("start_position") or closing_segment.get("begin", 0)

                begin_global = closing_offset + corrected_time["begin"]
                end_global = closing_offset + corrected_time["end"]

                entities.append({
                    "type": "HORARIO-FIM",
                    "begin": begin_global,
                    "end": end_global,
                    "text": corrected_time["text"],
                    "metadata_type": "Horário",
                    "attributes": {"Horrio": "fim"}
                })
            else:
                logger.warning(f"End time '{end_time_info['text']}' not found in {muni}")
        else:
            # Use global offsets provided in metadata
            entities.append({
                "type": "HORARIO-FIM",
                "begin": end_time_info["begin"],
                "end": end_time_info["end"],
                "text": end_time_info["text"],
                "metadata_type": "Horário",
                "attributes": {"Horrio": "fim"}
            })

    # === Location ===
    if metadata.get("location"):
        location_info = metadata["location"]
        entities.append({
            "type": "LOCAL",
            "begin": location_info["begin"],
            "end": location_info["end"],
            "text": location_info["text"],
            "metadata_type": "Local",
            "attributes": {}
        })

    # === Meeting type ===
    if metadata.get("meeting_type"):
        meeting_info = metadata["meeting_type"]
        meeting_text = meeting_info["text"].lower()

        if any(word in meeting_text for word in ["extraordinária", "extraordinaria", "extraordinary"]):
            entity_type = "TIPO-REUNIAO-EXTRAORDINARIA"
        elif any(word in meeting_text for word in ["ordinária", "ordinaria", "ordinary"]):
            entity_type = "TIPO-REUNIAO-ORDINARIA"
        else:
            entity_type = "TIPO-REUNIAO-ORDINARIA"  # Default fallback

        entities.append({
            "type": entity_type,
            "begin": meeting_info["begin"],
            "end": meeting_info["end"],
            "text": meeting_info["text"],
            "metadata_type": "Tipo de reunião",
            "attributes": {"TipodeReunio": meeting_text}
        })

    # === Minute ID (meeting number) ===
    if metadata.get("minute_id"):
        minute_info = metadata["minute_id"]
        entities.append({
            "type": "NUMERO-ATA",
            "begin": minute_info["begin"],
            "end": minute_info["end"],
            "text": minute_info["text"],
            "metadata_type": "Número da ata",
            "attributes": {}
        })

    # === Participants ===
    if metadata.get("participants"):
        for participant in metadata["participants"]:
            if "begin" not in participant or "end" not in participant:
                continue

            participant_type = participant.get("type", "").lower()
            present_status = participant.get("present", "present")

            # Determine participant entity type
            if any(word in participant_type for word in ["vice-president", "councillor", "vereador"]):
                entity_type = "PARTICIPANTE-VEREADOR"
            elif "president" in participant_type:
                entity_type = "PARTICIPANTE-PRESIDENTE"
            elif any(word in participant_type for word in ["staff", "funcionário", "public", "público"]):
                continue  # Ignore staff/public mentions
            else:
                entity_type = "PARTICIPANTE-VEREADOR"  # Default fallback

            entities.append({
                "type": entity_type,
                "begin": participant["begin"],
                "end": participant["end"],
                "text": participant["name"],
                "metadata_type": "Participantes",
                "attributes": {
                    "Participantes": participant_type,
                    "Presenca": present_status,
                    "Partido": participant.get("party", "")
                }
            })

    return entities


def generate_chunks_from_segment(segment_text, segment_offset, label, entities, ata_id):
    """
    Split a text segment (opening or closing) into overlapping chunks
    and align entities to their corresponding positions within each chunk.

    Args:
        segment_text (str): text of the segment (e.g., opening section)
        segment_offset (int): starting offset of the segment in the full document
        label (str): section label, e.g. 'intro' or 'ending'
        entities (list[dict]): list of all entities with global positions
        ata_id (str): document identifier

    Returns:
        list[dict]: list of chunk dictionaries with adjusted entity offsets.
    """
    if not segment_text.strip():
        return []

    # Use a sliding window approach for splitting large segments
    chunker = DocumentChunker(chunk_size=600, chunk_overlap=200)
    chunks = chunker.chunk_document(segment_text)
    result = []

    for chunk in chunks:
        # Determine chunk position inside the segment
        chunk_start_in_segment = segment_text.find(chunk)
        if chunk_start_in_segment == -1:
            continue
        chunk_end_in_segment = chunk_start_in_segment + len(chunk)

        chunk_entities = []

        # Align entities that overlap with this chunk
        for entity in entities:
            # Convert global positions to segment-relative
            entity_begin = entity["begin"] - segment_offset
            entity_end = entity["end"] - segment_offset

            # Check for overlap between entity and chunk
            if not (entity_end <= chunk_start_in_segment or entity_begin >= chunk_end_in_segment):
                # Adjust entity position to chunk-relative coordinates
                adjusted_begin = max(0, entity_begin - chunk_start_in_segment)
                adjusted_end = min(len(chunk), entity_end - chunk_start_in_segment)

                # Only include entities with valid overlap
                if adjusted_begin < adjusted_end:
                    chunk_entities.append({
                        "type": entity["type"],
                        "begin": adjusted_begin,
                        "end": adjusted_end,
                        "text": chunk[adjusted_begin:adjusted_end],
                        "metadata_type": entity["metadata_type"],
                        "attributes": entity["attributes"]
                    })

        # Save the chunk (even if no entities exist — ensures dataset balance)
        result.append({
            "text": chunk,
            "entities": chunk_entities,
            "ata_id": ata_id,
            "section": label
        })

    return result


def extract_metadata_entities(record):
    """
    Extract all metadata entities from a record in the new dataset format.
    Each record may contain multiple municipalities and multiple documents.

    Args:
        record (dict): structure like:
            {
              "documents": {
                "Alandroal": {
                    "doc_001": {"metadata": {...}},
                    "doc_002": {"metadata": {...}},
                    ...
                }
              }
            }

    Returns:
        list[dict]: list of processed document chunks (with aligned entities)
    """
    documents = []

    # Check expected nested structure
    if "documents" in record:
        for municipality_name, municipality_data in record["documents"].items():
            for document_id, document_data in municipality_data.items():
                metadata = document_data.get("metadata", {})

                # Inject municipality name into metadata (used later)
                metadata["municipality"] = municipality_name

                # Determine document ID
                ata_id = metadata.get("document_id", document_id)

                # Extract text segments from metadata
                opening_segment = metadata.get("opening_segment", {})
                closing_segment = metadata.get("closing_segment", {})

                opening_text = opening_segment.get("text", "")
                closing_text = closing_segment.get("text", "")

                if not opening_text and not closing_text:
                    logger.warning(f"No text found in document {ata_id}")
                    continue

                # Extract entity spans from metadata (may adjust end time using closing text)
                metadata_entities = extract_entities_from_metadata(metadata, closing_text)

                # --- Process opening segment ---
                if opening_text:
                    opening_offset = opening_segment.get("start_position") or opening_segment.get("begin", 0)
                    opening_chunks = generate_chunks_from_segment(
                        opening_text,
                        opening_offset,
                        "intro",
                        metadata_entities,
                        ata_id
                    )
                    documents.extend(opening_chunks)

                # --- Process closing segment ---
                if closing_text:
                    closing_offset = closing_segment.get("start_position") or closing_segment.get("begin", 0)
                    closing_chunks = generate_chunks_from_segment(
                        closing_text,
                        closing_offset,
                        "ending",
                        metadata_entities,
                        ata_id
                    )
                    documents.extend(closing_chunks)
    else:
        # Unexpected data structure
        logger.warning(f"Unknown record structure: {list(record.keys()) if isinstance(record, dict) else type(record)}")

    return documents


def mascarar_documento(doc):
    """
    Apply synthetic data masking to anonymize sensitive information
    (names, places, times, dates, municipalities, etc.).
    This creates a masked version of the text while preserving entity spans.

    Args:
        doc (dict): document dictionary with fields:
            {
              "text": str,
              "entities": list[dict]
            }

    Returns:
        dict: document with new masked text and updated entity offsets.
    """
    faker_pt = Faker("pt_PT")

    # List of municipalities used to detect and mask mentions
    MUNICIPIOS = [
        "Alandroal", "Campo Maior", "Covilhã", "Fundão", "Guimarães", "Porto",
        "Câmara Municipal de Alandroal", "Câmara Municipal de Campo Maior",
        "Município de Guimarães", "Concelho de Covilhã"
    ]

    original_text = doc["text"]
    entities = doc["entities"]
    nome_map = {}        # Keep track of fake names assigned to real names
    hora_inicio = None   # To maintain consistency between start/end times

    # === (1) Detect extra spans for municipality names ===
    municipio_spans = []
    spans_ocupados = [(e["begin"], e["end"]) for e in entities]

    for municipio in MUNICIPIOS:
        for match in re.finditer(re.escape(municipio), original_text, flags=re.IGNORECASE):
            b, e = match.start(), match.end()

            # Ignore overlaps with existing entity spans
            if any(not (e <= xb or b >= xe) for xb, xe in spans_ocupados):
                continue
            # Ignore overlaps with previously found municipalities
            if any(not (e <= xb or b >= xe) for xb, xe in [(m["begin"], m["end"]) for m in municipio_spans]):
                continue

            municipio_spans.append({
                "begin": b,
                "end": e,
                "type": "MUNICIPIO",
                "text": match.group()
            })

    # === (2) Combine real and artificial spans ===
    combined_spans = [
        {**ent, "source": "original"} for ent in entities
    ] + [
        {**span, "source": "municipio"} for span in municipio_spans
    ]
    combined_spans.sort(key=lambda e: e["begin"])

    # === (3) Build new text while replacing sensitive spans ===
    new_text_parts = []
    new_entities = []
    cursor = 0

    for ent in combined_spans:
        start, end = ent["begin"], ent["end"]
        original_span = original_text[start:end]
        new_text_parts.append(original_text[cursor:start])

        # Replace text depending on entity type
        if ent["source"] == "municipio":
            new_span = "@MUNICIPIO"

        elif ent["type"].startswith(("PARTICIPANTE-FUNCIONARIO", "PARTICIPANTE-PUBLICO")):
            if original_span not in nome_map:
                nome_map[original_span] = faker_pt.name()
            new_span = nome_map[original_span]

        elif ent["type"].startswith("PARTICIPANTE"):
            if random.random() < 0.6:  # 60% chance to anonymize
                if original_span not in nome_map:
                    nome_map[original_span] = faker_pt.name()
                new_span = nome_map[original_span]
            else:
                new_span = original_span

        elif ent["type"] == "LOCAL":
            new_span = faker_pt.address() if random.random() < 0.6 else original_span

        elif ent["type"] == "HORARIO-INICIO":
            # Randomly generate a plausible start time
            hora_inicio = random.randint(8, 17)
            minuto_inicio = random.choice([0, 15, 30, 45])
            if random.random() < 0.3:  # 30% chance to reformat
                formatos = [
                    f"{hora_inicio:02d}:{minuto_inicio:02d}",
                    f"{hora_inicio}h{minuto_inicio:02d}min",
                    f"{hora_inicio} horas e {minuto_inicio} minutos",
                    f"{hora_inicio:02d}.{minuto_inicio:02d}"
                ]
                new_span = random.choice(formatos)
            else:
                new_span = original_span

        elif ent["type"] == "HORARIO-FIM":
            # Keep logical consistency with start time if available
            hora_fim = min((hora_inicio or random.randint(9, 18)) + 1, 18)
            minuto_fim = random.choice([0, 15, 30, 45])
            if random.random() < 0.3:
                formatos = [
                    f"{hora_fim:02d}:{minuto_fim:02d}",
                    f"{hora_fim}h{minuto_fim:02d}min",
                    f"{hora_fim} horas e {minuto_fim} minutos",
                    f"{hora_fim:02d}.{minuto_fim:02d}"
                ]
                new_span = random.choice(formatos)
            else:
                new_span = original_span

        elif ent["type"] == "DATA":
            # Randomly reformat date strings
            data_obj = dateparser.parse(original_span, languages=['pt'])
            if data_obj and random.random() < 0.3:
                meses_pt = [
                    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
                    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"
                ]
                dia = data_obj.day
                mes_nome = meses_pt[data_obj.month - 1]
                ano = data_obj.year
                formatos = [
                    data_obj.strftime("%d/%m/%Y"),
                    data_obj.strftime("%d-%m-%Y"),
                    data_obj.strftime("%Y-%m-%d"),
                    f"{dia} de {mes_nome} de {ano}",
                    f"{dia} {mes_nome} {ano}",
                    data_obj.strftime("%d.%m.%Y")
                ]
                new_span = random.choice(formatos)
            else:
                new_span = original_span

        else:
            # Default: keep unchanged
            new_span = original_span

        # Compute new offsets after replacement
        new_begin = sum(len(p) for p in new_text_parts)
        new_end = new_begin + len(new_span)

        # Only keep entity annotation for original (not artificial) entities
        if ent["source"] == "original":
            new_entities.append({
                **ent,
                "begin": new_begin,
                "end": new_end,
                "text": new_span
            })

        # Append to text builder
        new_text_parts.append(new_span)
        cursor = end

    # Add trailing text after last entity
    new_text_parts.append(original_text[cursor:])
    new_text = "".join(new_text_parts)

    # Return new masked document
    return {
        **doc,
        "text": new_text,
        "entities": new_entities
    }


def create_token_classification_examples(documents, apply_masking=True):
    """
    Convert preprocessed documents (with entities and offsets)
    into token-level examples suitable for NER training.

    Args:
        documents (list[dict]): list of chunks returned by extract_metadata_entities()
        apply_masking (bool): if True, applies anonymization via mascarar_documento()

    Returns:
        list[dict]: token-level examples in the structure:
            {
              "tokens": [...],
              "tags": [...],
              "text": "...",
              "entities": [...],
              "ata_id": "...",
              "section": "intro"|"ending"
            }
    """
    examples = []

    for doc in documents:
        text = doc["text"]
        entities = doc["entities"]

        # Skip chunks without entities unless from the 'ending' section
        if not entities and doc.get("section") != "ending":
            continue

        # Optionally apply anonymization
        # if apply_masking:
        #     doc = mascarar_documento(doc)

        # Tokenize using custom tokenizer (handles dates/times correctly)
        tokens, token_positions = custom_tokenize_datetime(text)
        tags = ["O"] * len(tokens)

        # === Assign BIO tags based on entity spans ===
        for entity in entities:
            entity_begin = entity["begin"]
            entity_end = entity["end"]
            entity_type = entity["type"]

            # Normalize participant presence (Present/Ausente/Substituído)
            presenca = entity.get("attributes", {}).get("Presenca")
            if presenca:
                presenca_norm = presenca.strip().lower()
                if presenca_norm in ["presente", "present"]:
                    entity_type = f"{entity_type}-PRESENTE"
                elif presenca_norm in ["ausente", "absent"]:
                    entity_type = f"{entity_type}-AUSENTE"
                elif presenca_norm in ["substituido", "substituted"]:
                    entity_type = f"{entity_type}-SUBSTITUIDO"

            # Find tokens overlapping with entity span
            entity_tokens = []
            for i, token_pos in enumerate(token_positions):
                token_start = token_pos
                token_end = token_pos + len(tokens[i])
                # Overlap condition
                if not (token_end <= entity_begin or token_start >= entity_end):
                    entity_tokens.append(i)

            # Assign BIO tags
            if entity_tokens:
                tags[entity_tokens[0]] = f"B-{entity_type}"
                for token_idx in entity_tokens[1:]:
                    tags[token_idx] = f"I-{entity_type}"

        # Build training example
        examples.append({
            "text": text,
            "tokens": tokens,
            "tags": tags,
            "entities": entities,
            "ata_id": doc.get("ata_id"),
            "section": doc.get("section")
        })

    return examples

def process_municipality_data(municipality_name, municipality_data, apply_masking=True):
    """
    Process all documents for a single municipality.

    This function extracts entities, generates chunks, and creates
    token classification examples (optionally applying masking).

    Args:
        municipality_name (str): name of the municipality
        municipality_data (dict): full data structure for the municipality
        apply_masking (bool): whether to apply anonymization (for training only)

    Returns:
        list[dict]: token classification examples ready for saving.
    """
    logger.info(f"Processing municipality {municipality_name}")

    # Handle expected nested structure: { "documents": { "<Municipality>": {...} } }
    if isinstance(municipality_data, dict) and "documents" in municipality_data:
        municipality_docs = municipality_data["documents"].get(municipality_name, {})
        record = {"documents": {municipality_name: municipality_docs}}
    else:
        logger.error(f"Unexpected structure for {municipality_name}: {type(municipality_data)}")
        return []

    # Extract metadata-based entities and chunked segments
    documents = extract_metadata_entities(record)

    # Convert them into token classification examples (BIO format)
    examples = create_token_classification_examples(documents, apply_masking=apply_masking)
    return examples


def load_json_files_from_directory(directory_path):
    """
    Load all JSON files from a directory in the new structured format.

    Expected file structure:
    {
        "documents": {
            "<Municipio>": {
                "<document_id>": { "metadata": {...}, ... }
            }
        }
    }

    Args:
        directory_path (str): folder path containing municipality JSON files.

    Returns:
        dict: { municipality_name: municipality_data }
    """
    json_files = {}
    directory = Path(directory_path)

    for json_file in directory.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate format
            if not isinstance(data, dict) or "documents" not in data:
                logger.error(f"{json_file.name} does not match expected format (missing 'documents')")
                continue

            # Retrieve municipality name and count meetings
            municipality_name = list(data["documents"].keys())[0]
            total_meetings = len(data["documents"][municipality_name])
            logger.info(f"Loaded {json_file.name}: {municipality_name} ({total_meetings} meetings)")

            json_files[municipality_name] = data

        except Exception as e:
            logger.error(f"Error loading {json_file.name}: {str(e)}")

    return json_files


def main():
    """
    Main execution function for cross-validation dataset preparation.

    Steps:
      1. Load all municipality files (structured JSON format)
      2. Load train/val/test splits from splits.json
      3. Create one fold per municipality (leave-one-out setup)
      4. Save train/val/test examples in JSONL format per fold
    """
    parser = argparse.ArgumentParser(description="Process metadata annotations from new dataset format")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="ecir_submission/data/dataset_metadata_pt",
        help="Path to directory containing structured JSON files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ecir_submission/data/metadata_final",
        help="Output directory for processed cross-validation datasets"
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # === Load all JSON files per municipality ===
    municipality_data = load_json_files_from_directory(args.input_dir)
    municipality_names = list(municipality_data.keys())

    # === Load split definitions (train/val/test per document) ===
    with open("ecir_submission/data/split/splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    train_files = set(splits["train_files"])
    val_files = set(splits["val_files"])
    test_files = set(splits["test_files"])

    # === Create leave-one-out folds ===
    for i, test_municipality in enumerate(municipality_names):
        dataset_name = f"fold_{i + 1}_{test_municipality.lower()}_test"
        dataset_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        all_train_examples, all_val_examples = [], []

        # Train on all municipalities except the test one
        for train_muni in municipality_names:
            if train_muni == test_municipality:
                continue

            # Process and mask training data
            muni_examples = process_municipality_data(
                train_muni,
                municipality_data[train_muni],
                apply_masking=True
            )

            # Split examples into train/val using splits.json
            for ex in muni_examples:
                ata_id = f"{ex['ata_id']}.json"
                if ata_id in train_files or ata_id in val_files:
                    all_train_examples.append(ex)
                elif ata_id in test_files:
                    all_val_examples.append(ex)

        # Test set: full municipality, without masking
        all_test_examples = process_municipality_data(
            test_municipality,
            municipality_data[test_municipality],
            apply_masking=False
        )

        # === Save datasets to disk in JSONL format ===
        train_file = os.path.join(dataset_dir, "train.jsonl")
        test_file = os.path.join(dataset_dir, "test.jsonl")
        val_file = os.path.join(dataset_dir, "val.jsonl")

        for path, examples in [(train_file, all_train_examples),
                               (test_file, all_test_examples),
                               (val_file, all_val_examples)]:
            with open(path, "w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        logger.info(
            f"Saved fold {i + 1}: "
            f"train={len(all_train_examples)} | "
            f"val={len(all_val_examples)} | "
            f"test={len(all_test_examples)} → {dataset_dir}"
        )


if __name__ == "__main__":
    main()
