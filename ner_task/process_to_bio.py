"""
Process Metadata Annotations - New Format

This script processes metadata from the new structured dataset format
and converts them into the JSONL format used for token classification training.
It extracts metadata entities from the opening and closing segments of documents
and generates BIO-tagged examples for NER training.

Default language: Portuguese (spaCy model "pt_core_news_sm")
→ To use English, simply change the model variable to "en_core_web_sm".
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

# -----------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# Language model configuration
# -----------------------------------------------------------
# Default: Portuguese model
# To switch to English, replace with "en_core_web_sm"
model = "pt_core_news_sm"

# Load spaCy model for tokenization and sentence splitting
try:
    nlp = spacy.load(model)
    logger.info(f"Loaded spaCy model: {model}")
except Exception:
    logger.warning(f"Could not load spaCy model. Installing...")
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", model])
    nlp = spacy.load(model)
    logger.info(f"Installed and loaded spaCy model")


# -----------------------------------------------------------
# Utility: Tokenization of dates and times
# -----------------------------------------------------------
def custom_tokenize_datetime(text):
    """
    Custom tokenizer that properly splits date and time patterns within text.
    Example: "11/09/2024" → ["11", "/", "09", "/", "2024"]
             "16:00" → ["16", ":", "00"]
    """
    pattern = r'(\d{1,2}/\d{1,2}/\d{4})|(\d{1,2}[:.]\d{2})'
    matches = list(re.finditer(pattern, text))

    tokens = []
    token_positions = []
    last_end = 0

    for match in matches:
        start, end = match.span()

        # Tokens before the matched expression
        if start > last_end:
            pre_text = text[last_end:start]
            pre_doc = nlp(pre_text)
            for token in pre_doc:
                tokens.append(token.text)
                token_positions.append(last_end + token.idx)

        matched_text = match.group()

        # Split date tokens
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
        # Split time tokens
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

    # Tokens after last match
    if last_end < len(text):
        post_text = text[last_end:]
        post_doc = nlp(post_text)
        for token in post_doc:
            tokens.append(token.text)
            token_positions.append(last_end + token.idx)

    return tokens, token_positions


# -----------------------------------------------------------
# End time correction in closing segments
# -----------------------------------------------------------
def find_end_time_in_closing(closing_text, metadata_end_time_text):
    """
    Search for the end time expression inside the closing segment text.
    Returns its relative offsets within the segment.
    """
    if not closing_text or not metadata_end_time_text:
        return None

    idx = closing_text.lower().find(metadata_end_time_text.lower())
    if idx != -1:
        return {
            "text": metadata_end_time_text,
            "start": idx,
            "end": idx + len(metadata_end_time_text)
        }

    # Fallback to pattern-based search
    time_patterns = [
        r'\b\d{1,2}[:.]\d{2}\s*horas?\b',
        r'\b\d{1,2}\s*horas?\s*e\s*\d{1,2}\s*minutos?\b'
    ]

    for pattern in time_patterns:
        matches = list(re.finditer(pattern, closing_text, re.IGNORECASE))
        if matches:
            match = matches[-1]  # last one is likely the final time
            return {
                "text": match.group(),
                "start": match.start(),
                "end": match.end()
            }

    return None


def extract_entities_from_metadata(metadata, closing_text=""):
    """
    Extract structured metadata entities (e.g., date, times, participants)
    from the nested metadata dictionary of each document.
    Optionally uses the closing segment text for refining end time offsets.
    """
    entities = []

    # ---- Date ----
    if "date" in metadata and metadata["date"]:
        date_info = metadata["date"]
        entities.append({
            "type": "DATA",
            "begin": date_info.get("start", date_info.get("begin", 0)),
            "end": date_info.get("end", 0),
            "text": date_info["text"],
            "metadata_type": "Data",
            "attributes": {}
        })

    # ---- Begin time ----
    if "begin_time" in metadata and metadata["begin_time"]:
        time_info = metadata["begin_time"]
        entities.append({
            "type": "HORARIO-INICIO",
            "begin": time_info.get("start", time_info.get("begin", 0)),
            "end": time_info.get("end", 0),
            "text": time_info["text"],
            "metadata_type": "Horário",
            "attributes": {"Horrio": "início"}
        })

    # ---- End time ----
    if "end_time" in metadata and metadata["end_time"]:
        end_time_info = metadata["end_time"]
        muni = metadata.get("municipality", "").lower()

        # In some municipalities, the end time is located inside the closing segment
        closing_munis = ["alandroal", "campomaior", "guimaraes"]

        if any(m in muni for m in closing_munis) and closing_text:
            # Try to locate the end time within the closing text
            corrected_time = find_end_time_in_closing(closing_text, end_time_info["text"])
            if corrected_time:
                # Convert relative offsets (within segment) to global document offsets
                closing_segment = metadata.get("closing_segment", {})
                closing_offset = closing_segment.get("start", closing_segment.get("begin", 0))
                begin_global = closing_offset + corrected_time["start"]
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
                logger.warning(f"Could not locate end time '{end_time_info['text']}' in {muni}")
        else:
            # Use original offsets if no correction is needed
            entities.append({
                "type": "HORARIO-FIM",
                "begin": end_time_info.get("start", end_time_info.get("begin", 0)),
                "end": end_time_info.get("end", 0),
                "text": end_time_info["text"],
                "metadata_type": "Horário",
                "attributes": {"Horrio": "fim"}
            })

    # ---- Location ----
    if "location" in metadata and metadata["location"]:
        location_info = metadata["location"]
        entities.append({
            "type": "LOCAL",
            "begin": location_info.get("start", location_info.get("begin", 0)),
            "end": location_info.get("end", 0),
            "text": location_info["text"],
            "metadata_type": "Local",
            "attributes": {}
        })

    # ---- Meeting type ----
    if "meeting_type" in metadata and metadata["meeting_type"]:
        meeting_info = metadata["meeting_type"]
        meeting_text = meeting_info["text"].lower()

        # Normalize entity type according to meeting type
        if "extraordinária" in meeting_text or "extraordinary" in meeting_text:
            entity_type = "TIPO-REUNIAO-EXTRAORDINARIA"
        elif "ordinária" in meeting_text or "ordinary" in meeting_text:
            entity_type = "TIPO-REUNIAO-ORDINARIA"
        else:
            entity_type = "TIPO-REUNIAO-ORDINARIA"  # default fallback

        entities.append({
            "type": entity_type,
            "begin": meeting_info.get("start", meeting_info.get("begin", 0)),
            "end": meeting_info.get("end", 0),
            "text": meeting_info["text"],
            "metadata_type": "Tipo de reunião",
            "attributes": {"TipodeReunio": meeting_text}
        })

    # ---- Minute ID (meeting number) ----
    if "minute_id" in metadata and metadata["minute_id"]:
        minute_info = metadata["minute_id"]
        entities.append({
            "type": "NUMERO-ATA",
            "begin": minute_info.get("start", minute_info.get("begin", 0)),
            "end": minute_info.get("end", 0),
            "text": minute_info["text"],
            "metadata_type": "Número da ata",
            "attributes": {}
        })

    # ---- Participants ----
    if "participants" in metadata and metadata["participants"]:
        for participant in metadata["participants"]:
            part_start = participant.get("start", participant.get("begin"))
            if part_start is None or "end" not in participant:
                continue

            participant_type = participant.get("type", "").lower()
            present = participant.get("present", "present")

            # Map textual participant type to normalized label
            if "vice-president" in participant_type or "vereador" in participant_type:
                entity_type = "PARTICIPANTE-VEREADOR"
            elif "president" in participant_type:
                entity_type = "PARTICIPANTE-PRESIDENTE"
            elif "staff" in participant_type or "funcionário" in participant_type:
                continue  # skip administrative staff
            elif "public" in participant_type or "público" in participant_type:
                continue  # skip public attendees
            else:
                entity_type = "PARTICIPANTE-VEREADOR"  # default fallback

            entities.append({
                "type": entity_type,
                "begin": part_start,
                "end": participant["end"],
                "text": participant["name"],
                "metadata_type": "Participantes",
                "attributes": {
                    "Participantes": participant_type,
                    "Presenca": present,
                    "Partido": participant.get("party", "")
                }
            })

    return entities


def generate_chunks_from_segment(segment_text, segment_offset, label, entities, ata_id):
    """
    Divide a text segment (opening or closing) into smaller overlapping chunks
    for training efficiency. Each chunk inherits only the entities overlapping
    with its character span, with adjusted offsets.
    """
    if not segment_text.strip():
        return []

    # Create chunker with 600-char chunks and 200-char overlap
    chunker = DocumentChunker(chunk_size=600, chunk_overlap=200)
    chunks = chunker.chunk_document(segment_text)
    result = []

    for chunk in chunks:
        chunk_start_in_segment = segment_text.find(chunk)
        if chunk_start_in_segment == -1:
            continue

        chunk_end_in_segment = chunk_start_in_segment + len(chunk)
        chunk_entities = []

        # Identify all entities that overlap with this chunk
        for entity in entities:
            entity_begin = entity["begin"] - segment_offset
            entity_end = entity["end"] - segment_offset

            # Check if entity overlaps with current chunk
            if not (entity_end <= chunk_start_in_segment or entity_begin >= chunk_end_in_segment):
                adjusted_begin = max(0, entity_begin - chunk_start_in_segment)
                adjusted_end = min(len(chunk), entity_end - chunk_start_in_segment)

                if adjusted_begin < adjusted_end:
                    chunk_entities.append({
                        "type": entity["type"],
                        "begin": adjusted_begin,
                        "end": adjusted_end,
                        "text": chunk[adjusted_begin:adjusted_end],
                        "metadata_type": entity["metadata_type"],
                        "attributes": entity["attributes"]
                    })

        # Add chunk to the result, even if no entities were found
        result.append({
            "text": chunk,
            "entities": chunk_entities,
            "ata_id": ata_id,
            "section": label
        })

    return result


def extract_metadata_entities(record):
    """
    Extract metadata entities from the new structured dataset format.

    Each record may contain multiple documents, organized by municipality.
    This function iterates through all municipalities and documents, retrieves
    metadata fields (e.g., opening/closing segments), and extracts entities
    for Named Entity Recognition (NER) training.
    """
    documents = []

    # Handle nested structure: record["documents"] -> {municipality -> {document_id -> metadata}}
    if "documents" in record:
        for municipality_name, municipality_data in record["documents"].items():
            for document_id, document_data in municipality_data.items():
                metadata = document_data.get("metadata", {})

                # Inject municipality name into metadata for contextual reference
                metadata["municipality"] = municipality_name

                # Document unique identifier (fallback to dictionary key)
                ata_id = metadata.get("document_id", document_id)

                # Extract opening and closing text segments
                opening_segment = metadata.get("opening_segment", {})
                closing_segment = metadata.get("closing_segment", {})

                opening_text = opening_segment.get("text", "")
                closing_text = closing_segment.get("text", "")

                if not opening_text and not closing_text:
                    logger.warning(f"No text found in document {ata_id}")
                    continue

                # Extract all metadata-based entities (dates, times, participants, etc.)
                metadata_entities = extract_entities_from_metadata(metadata, closing_text)

                # ---- Opening segment ----
                if opening_text:
                    opening_offset = opening_segment.get("start", opening_segment.get("begin", 0))
                    opening_chunks = generate_chunks_from_segment(
                        opening_text, opening_offset, "intro", metadata_entities, ata_id
                    )
                    documents.extend(opening_chunks)

                # ---- Closing segment ----
                if closing_text:
                    closing_offset = closing_segment.get("start", closing_segment.get("begin", 0))
                    closing_chunks = generate_chunks_from_segment(
                        closing_text, closing_offset, "ending", metadata_entities, ata_id
                    )
                    documents.extend(closing_chunks)

    else:
        # If the input format is not as expected, log the available structure
        logger.warning(f"Unknown record structure: {list(record.keys()) if isinstance(record, dict) else type(record)}")

    return documents


def mascarar_documento(doc):
    """
    Apply data masking / deslexicalization to protect sensitive information
    (e.g., names, places, times, dates) and to improve model generalization.

    Uses the 'faker' library to generate synthetic replacements for
    participant names, municipalities, and addresses, while preserving
    text structure and entity offsets.
    """
    faker_pt = Faker("pt_PT")

    # Predefined list of known municipalities for masking
    MUNICIPIOS = [
        "Alandroal", "Campo Maior", "Covilhã", "Fundão", "Guimarães", "Porto",
        "Câmara Municipal de Alandroal", "Câmara Municipal de Campo Maior",
        "Município de Guimarães", "Concelho de Covilhã"
    ]

    original_text = doc["text"]
    entities = doc["entities"]
    nome_map = {}  # map original → fake names (keeps consistency)
    hora_inicio = None  # store start hour to keep coherence with end time

    # ---------------------------------------------------------
    # 1. Identify synthetic spans for municipality mentions
    # ---------------------------------------------------------
    municipio_spans = []
    spans_ocupados = [(e["begin"], e["end"]) for e in entities]

    for municipio in MUNICIPIOS:
        for match in re.finditer(re.escape(municipio), original_text, flags=re.IGNORECASE):
            b, e = match.start(), match.end()

            # Skip if overlaps with any existing entity
            sobrepoe = any(not (e <= xb or b >= xe) for xb, xe in spans_ocupados)
            if sobrepoe:
                continue

            # Skip if overlaps with already detected municipality span
            sobrepoe = any(not (e <= xb or b >= xe) for xb, xe in [(m["begin"], m["end"]) for m in municipio_spans])
            if sobrepoe:
                continue

            municipio_spans.append({
                "begin": b,
                "end": e,
                "type": "MUNICIPIO",
                "text": match.group()
            })

    # ---------------------------------------------------------
    # 2. Merge all spans (original + synthetic) and sort by position
    # ---------------------------------------------------------
    combined_spans = [
        {**ent, "source": "original"} for ent in entities
    ] + [
        {**span, "source": "municipio"} for span in municipio_spans
    ]
    combined_spans.sort(key=lambda e: e["begin"])

    # ---------------------------------------------------------
    # 3. Rebuild text with masked replacements and updated offsets
    # ---------------------------------------------------------
    new_text_parts = []
    new_entities = []
    cursor = 0

    for ent in combined_spans:
        start, end = ent["begin"], ent["end"]
        original_span = original_text[start:end]

        # Keep original text between entities
        new_text_parts.append(original_text[cursor:start])

        # ---- Replace entity text depending on its type ----
        if ent["source"] == "municipio":
            new_span = "@MUNICIPIO"

        elif ent["type"].startswith("PARTICIPANTE-FUNCIONARIO") or ent["type"].startswith("PARTICIPANTE-PUBLICO"):
            # Replace staff or public names with synthetic names
            if original_span not in nome_map:
                nome_map[original_span] = faker_pt.name()
            new_span = nome_map[original_span]

        elif ent["type"].startswith("PARTICIPANTE"):
            # 60% chance to replace participant names with fake ones
            if random.random() < 0.6:
                if original_span not in nome_map:
                    nome_map[original_span] = faker_pt.name()
                new_span = nome_map[original_span]
            else:
                new_span = original_span

        elif ent["type"] == "LOCAL":
            # Replace addresses with synthetic ones (60% chance)
            if random.random() < 0.6:
                new_span = faker_pt.address()
            else:
                new_span = original_span

        elif ent["type"] == "HORARIO-INICIO":
            # Generate random realistic start time
            hora_inicio = random.randint(8, 17)
            minuto_inicio = random.choice([0, 15, 30, 45])
            if random.random() < 0.3:
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
            # Generate random coherent end time after start time
            if hora_inicio is None:
                hora_fim = random.randint(9, 18)
            else:
                hora_fim = min(hora_inicio + 1, 18)
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
            # Randomly reformat dates while keeping valid structure
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
            # Default: keep text unchanged
            new_span = original_span

        # Compute new character offsets based on rebuilt text
        new_begin = sum(len(p) for p in new_text_parts)
        new_end = new_begin + len(new_span)

        # Only reinsert original (not synthetic) entities
        if ent["source"] == "original":
            new_entities.append({
                **ent,
                "begin": new_begin,
                "end": new_end,
                "text": new_span
            })

        # Append replaced span and move cursor
        new_text_parts.append(new_span)
        cursor = end

    # Add remaining text after last entity
    new_text_parts.append(original_text[cursor:])
    new_text = "".join(new_text_parts)

    # Return masked document with updated entities
    return {
        **doc,
        "text": new_text,
        "entities": new_entities
    }


def create_token_classification_examples(documents, apply_masking=True):
    """
    Convert processed documents into token classification examples
    compatible with transformer-based NER models.

    Each document becomes a training instance with:
    - tokens (from spaCy or custom tokenizer)
    - BIO tags (B- / I- / O-)
    - metadata entities
    - document identifiers and section type

    Optionally applies masking/deslexicalization to protect sensitive info.
    """
    examples = []

    for doc in documents:
        # Apply masking only for training data (not validation/test)
        if apply_masking:
            doc = mascarar_documento(doc)

        text = doc["text"]
        entities = doc["entities"]

        # Skip empty examples unless they belong to the 'ending' section
        if not entities and doc.get("section") != "ending":
            continue

        # Tokenize text, splitting correctly around time/date expressions
        tokens, token_positions = custom_tokenize_datetime(text)
        tags = ["O"] * len(tokens)

        # --------------------------------------------------------
        # Map entities to tokens → assign BIO (Begin / Inside / Outside) tags
        # --------------------------------------------------------
        for entity in entities:
            entity_begin = entity["begin"]
            entity_end = entity["end"]
            entity_type = entity["type"]

            # Handle presence status (e.g., PRESENTE, AUSENTE)
            presenca = entity.get("attributes", {}).get("Presenca")
            if presenca:
                presenca_norm = presenca.strip().lower()
                if presenca_norm in ["presente", "present"]:
                    entity_type = f"{entity_type}-PRESENTE"
                elif presenca_norm in ["ausente", "absent"]:
                    entity_type = f"{entity_type}-AUSENTE"
                elif presenca_norm in ["substituido", "substituted"]:
                    entity_type = f"{entity_type}-SUBSTITUIDO"

            # Identify token indices overlapping with the entity
            entity_tokens = []
            for i, token_pos in enumerate(token_positions):
                if i < len(tokens):
                    token_start = token_pos
                    token_end = token_pos + len(tokens[i])

                    # Check if token overlaps with entity span
                    if not (token_end <= entity_begin or token_start >= entity_end):
                        entity_tokens.append(i)

            # Assign BIO tags to overlapping tokens
            if entity_tokens:
                for i, token_idx in enumerate(entity_tokens):
                    if i == 0:
                        tags[token_idx] = f"B-{entity_type}"
                    else:
                        tags[token_idx] = f"I-{entity_type}"

        # --------------------------------------------------------
        # Create a complete example for this document
        # --------------------------------------------------------
        example = {
            "text": text,
            "tokens": tokens,
            "tags": tags,
            "entities": entities,
            "ata_id": doc.get("ata_id"),
            "section": doc.get("section")
        }
        examples.append(example)

    return examples


def split_documents_by_filelist(documents, train_files, val_files, test_files):
    """
    Split all processed documents into train/validation/test sets
    based on file lists loaded from the split configuration.
    """
    train_docs, val_docs, test_docs = [], [], []

    for doc in documents:
        ata_id = doc.get("ata_id")
        if f"{ata_id}.json" in train_files:
            train_docs.append(doc)
        elif f"{ata_id}.json" in val_files:
            val_docs.append(doc)
        elif f"{ata_id}.json" in test_files:
            test_docs.append(doc)

    return train_docs, val_docs, test_docs


def load_json_files_from_directory(directory_path):
    """
    Load all JSON files from the input directory.

    Each file represents a municipality and must follow the "new format",
    where metadata is nested under a top-level 'documents' key.
    Logs the number of meeting minutes per municipality.
    """
    json_files = {}
    directory = Path(directory_path)

    for json_file in directory.glob("*.json"):
        municipality_name = json_file.stem.split("_")[0]

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # New dataset format must have the key "documents"
            if isinstance(data, dict) and "documents" in data:
                json_files[municipality_name] = data
                total_meetings = sum(len(v) for v in data["documents"].values())
                logger.info(f"{json_file.name}: {total_meetings} meetings")
            else:
                logger.warning(f"{json_file.name}: missing 'documents' key - skipping")
                continue

        except Exception as e:
            logger.error(f"Error loading {json_file}: {str(e)}")

    return json_files


def process_dataset_records(records, train_files, val_files, test_files, municipality_name=None):
    """
    Main processing function for each municipality.

    Converts all documents into BIO-tagged examples, split by dataset partition.
    Returns train, validation, and test examples ready for model training.
    """
    all_documents = []

    logger.info(f"Processing {len(records)} records for {municipality_name}")

    # Extract metadata entities and chunked segments for all records
    for record in tqdm(records, desc=f"Processing {municipality_name}"):
        try:
            docs = extract_metadata_entities(record)
            all_documents.extend(docs)
        except Exception as e:
            record_id = record.get("meeting_id", "unknown")
            logger.error(f"Error processing record {record_id}: {str(e)}")

    logger.info(f"Total documents extracted: {len(all_documents)}")

    # Split by predefined file lists
    train_docs, val_docs, test_docs = split_documents_by_filelist(
        all_documents, train_files, val_files, test_files
    )
    logger.info(f"Train documents: {len(train_docs)}, Val documents: {len(val_docs)}, Test documents: {len(test_docs)}")

    # Create examples (train with masking; val/test without masking)
    train_examples = create_token_classification_examples(train_docs, apply_masking=True)
    val_examples = create_token_classification_examples(val_docs, apply_masking=False)
    test_examples = create_token_classification_examples(test_docs, apply_masking=False)

    # Deduplication to avoid duplicated examples (same text and id)
    def deduplicate_examples(examples):
        seen = set()
        deduplicated = []
        for ex in examples:
            key = (ex["ata_id"], ex["text"])
            if key not in seen:
                seen.add(key)
                deduplicated.append(ex)
        return deduplicated

    train_examples = deduplicate_examples(train_examples)
    val_examples = deduplicate_examples(val_examples)
    test_examples = deduplicate_examples(test_examples)

    logger.info(f"Final - Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")

    return train_examples, val_examples, test_examples

def main():
    """Main execution function — orchestrates the full dataset processing pipeline."""
    parser = argparse.ArgumentParser(description="Process metadata annotations from new dataset format")
    parser.add_argument("--input_dir", type=str, default="ecir_submission/dataset_metadata_pt",
                        help="Path to directory containing JSON files")
    parser.add_argument("--output_dir", type=str, default="ecir_submission/data/metadata_final",
                        help="Output directory for processed datasets")
    parser.add_argument("--train_count", type=int, default=16,
                        help="Number of documents for training per municipality")
    parser.add_argument("--test_count", type=int, default=4,
                        help="Number of documents for testing per municipality")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset splits (train/val/test)
    with open("ecir_submission/split/splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    train_files = set(splits["train_files"])
    val_files = set(splits["val_files"])
    test_files = set(splits["test_files"])

    # Prepare output directories
    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")
    val_dir = os.path.join(args.output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Load input JSON files for each municipality
    municipality_data = load_json_files_from_directory(args.input_dir)
    if not municipality_data:
        logger.error("No JSON files found in the input directory")
        return

    logger.info(f"Found {len(municipality_data)} municipalities: {list(municipality_data.keys())}")

    # Initialize accumulators for combined datasets
    all_train_examples = []
    all_test_examples = []
    all_val_examples = []

    # ------------------------------------------------------------
    # Process each municipality individually
    # ------------------------------------------------------------
    for municipality_name, data in municipality_data.items():
        logger.info(f"\n=== Processing {municipality_name} ===")

        # Extract and convert data for this municipality
        train_examples, val_examples, test_examples = process_dataset_records(
            [data], train_files, val_files, test_files, municipality_name
        )

        # Normalize file naming (lowercase + accent removal)
        municipality_clean = municipality_name.lower().replace(" ", "_").replace("ç", "c").replace("ã", "a")

        # ---- Save per-municipality outputs ----
        if train_examples:
            train_file = os.path.join(train_dir, f"{municipality_clean}_train.jsonl")
            with open(train_file, "w", encoding="utf-8") as f:
                for example in train_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(train_examples)} train examples to {train_file}")

        if test_examples:
            test_file = os.path.join(test_dir, f"{municipality_clean}_test.jsonl")
            with open(test_file, "w", encoding="utf-8") as f:
                for example in test_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(test_examples)} test examples to {test_file}")

        if val_examples:
            val_file = os.path.join(val_dir, f"{municipality_clean}_val.jsonl")
            with open(val_file, "w", encoding="utf-8") as f:
                for example in val_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(val_examples)} val examples to {val_file}")

        # Accumulate examples for combined global dataset
        all_train_examples.extend(train_examples)
        all_val_examples.extend(val_examples)
        all_test_examples.extend(test_examples)

    # ------------------------------------------------------------
    # Save combined (all-municipality) datasets
    # ------------------------------------------------------------
    if all_train_examples:
        combined_train_file = os.path.join(args.output_dir, "combined_train.jsonl")
        with open(combined_train_file, "w", encoding="utf-8") as f:
            for example in all_train_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(all_train_examples)} combined train examples to {combined_train_file}")

    if all_test_examples:
        combined_test_file = os.path.join(args.output_dir, "combined_test.jsonl")
        with open(combined_test_file, "w", encoding="utf-8") as f:
            for example in all_test_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(all_test_examples)} combined test examples to {combined_test_file}")

    if all_val_examples:
        combined_val_file = os.path.join(args.output_dir, "combined_val.jsonl")
        with open(combined_val_file, "w", encoding="utf-8") as f:
            for example in all_val_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(all_val_examples)} combined val examples to {combined_val_file}")

    logger.info("Processing completed successfully!")

    # ------------------------------------------------------------
    # Final summary and entity statistics
    # ------------------------------------------------------------
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Total municipalities processed: {len(municipality_data)}")
    logger.info(f"Total train examples: {len(all_train_examples)}")
    logger.info(f"Total test examples: {len(all_test_examples)}")

    # Count total entity occurrences in training data
    entity_counts = {}
    for example in all_train_examples:
        for entity in example["entities"]:
            entity_type = entity["type"]
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

    logger.info("\nEntity counts in training set:")
    for entity_type, count in sorted(entity_counts.items()):
        logger.info(f"  {entity_type}: {count}")



if __name__ == "__main__":
    main()