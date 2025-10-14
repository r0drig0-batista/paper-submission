# MiNER: A Two-Stage Pipeline for Metadata Extraction from Municipal Meeting Minutes

A two-stage framework for **automatic metadata extraction** from municipal meeting minutes, combining **Question Answering (QA)** for segment boundary detection and **Named Entity Recognition (NER)** for fine-grained metadata extraction.  
This repository supports the experiments presented in the accompanying paper, providing all code, data splits. The trained models are available on Hugging Face.

---

## 1. Project Overview

This project introduces a unified pipeline for identifying and structuring metadata information in municipal meeting minutes.  
The pipeline operates in two stages:

1. **QA-based segmentation** – detects the *opening* and *closing* segments of each document.  
2. **NER-based metadata extraction** – extracts structured metadata fields such as date, location, meeting number and type, start and end times, and participants (as well as their presence).

Together, these components enable large-scale analysis of municipal records and serve as a benchmark for information extraction in long, formal administrative texts.

---

## 2. Key Features

- **Two-Stage Architecture:** Combines QA-based document segmentation with Transformer-based NER.  
- **Bilingual Dataset:** Portuguese originals and English translations from six municipalities.  
- **Structured Metadata Extraction:** Supports the following entities `minute_id`, `date`, `location`, `meeting_type`, `participants`, `begin_time`, and `end_time`.    
- **Efficiency & Performance:** Fine-tuned models outperform large generative LLMs while being orders of magnitude faster and greener.  
- **Open Resources:** All datasets and code are released for reproducibility.

---

## 3. Project Status

The core components of the MiNER framework are **fully implemented and validated**.
The system is considered **stable for research use**, and the codebase is actively maintained to ensure reproducibility.  
Minor improvements and refactoring are ongoing, particularly concerning dataset expansion and model evaluation consistency.

---

## 4. Technology Stack

### Language
- **Python 3.10+**

### Core Frameworks
- **PyTorch** – Deep learning backend for model training and inference  
- **Transformers (Hugging Face)** – Pre-trained language models for QA and NER  
- **Datasets & Evaluate (Hugging Face)** – Dataset management, preprocessing, and metric computation  
- **spaCy** – Sentence segmentation and preprocessing for QA dataset creation  
- **Faker** – Lexical masking and synthetic data generation for data augmentation  

### Utilities
- **NumPy** – Numerical operations and data manipulation  
- **LangChain** – Text chunking and utility functions for preprocessing  
- **tqdm** – Progress tracking and visualization during training  

### Development Tools
- **Git** – Version control and collaboration  
- **JSON** – Data serialization and storage format  
- **Markdown** – Documentation and reporting

---

## 5. Dependencies

All dependencies required to reproduce the experiments are listed in the `requirements.txt` file.  
The main libraries and minimum versions are:

transformers >= 4.30.0
datasets >= 2.14.0
torch >= 2.0.0
evaluate >= 0.4.0
spacy >= 3.5.0
faker >= 19.0.0
langchain >= 0.1.0
numpy >= 1.24.0
tqdm >= 4.65.0

To install all dependencies:

```bash
pip install -r requirements.txt

Installation
Prerequisites

    Python 3.10 or higher
    CUDA-capable GPU (recommended, but CPU mode is supported)
    At least 8GB RAM (16GB recommended for training)

Setup Steps

    Clone the repository  

    Create and activate a virtual environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

    Install dependencies

pip install -r requirements.txt
```

---

## 7. Usage

This section describes how to build datasets, train models, and reproduce the results for both stages of the MiNER pipeline.

---

### 7.1 Question Answering (QA)

**Goal:** Detect the *opening* and *closing* segments of each meeting minute, where metadata is typically concentrated.

#### Build the QA Dataset

```bash
python3 build_qa_dataset.py --lang pt
python3 build_qa_dataset.py --lang en
```

Train Model

```bash
python3 train_qa.py \
  --lang pt \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --fp16
```

Models:

- BERTimbau-Large

- XLM-RoBERTa-Large

Named Entity Recognition (NER)

Goal: Extract structured metadata entities from minutes.

Convert Metadata → BIO

```bash
python3 process_to_bio.py \
  --input_dir data/dataset_metadata_pt \
  --output_dir data/metadata_final
```
  
Tokenize & Align Labels

```bash
python3 transform_dataset.py
```

Train NER Model

```bash
python3 model.py
```


8. Dataset Description

Overview

Dataset Name: 				Council Metadata Corpus
Languages: 				Portuguese and English
Documents: 				6 municipalities × 20 minutes (2021–2024)
Total Tokens:				1,170,417 tokens
Total Tokens in Metadata Segments : 	32,364 tokens
Total Metadata Segments: 		180 segments
Annotation Fields:

date, minute_id, meeting_type, begin_time, end_time, location, participants (with role, presence, and party)

opening_segment, closing_segment

Format:
Each file (dataset_metadata_[lang]/municipality.json) contains:

```bash
{
  "documents": {
    "Municipality_Name": {
      "Municipality_cm_XXX_YYYY-MM-DD": {
        "document_id": "...",
        "full_text": "...",
        "metadata": {
          "minute_id": {...},
          "date": {...},
          "location": {...},
          "meeting_type": {...},
          "begin_time": {...},
          "end_time": {...},
          "participants": [...],
          "opening_segment": {...},
          "closing_segment": {...}
        }
      }
    }
  }
}
```

## 8. Dataset Description

### Data Files

The data files for the **Council Metadata Corpus** are located in the `data/` directory:

  -  dataset_metadata_en — Portuguese version (6 files with 20 documents each)
  -  dataset_metadata_pt — English version (6 files with 20 documents each)
  -  split — Train/val/test split information


Each JSON file corresponds to one municipality and contains the full text of the meeting minute, along with manually annotated metadata fields.

---

### Annotation Process

- **Source:** Official municipal meeting minutes provided by the respective municipalities (2021–2024)  
- **Annotation Tool:** [INCEpTION](https://inception-project.github.io/)  
- **Annotation Guidelines:** Annotators labeled only mentions of those metadata fields

---

### Dataset Characteristics

#### Challenges
- **Domain Specificity:** Contains formal administrative language and municipality-specific jargon.  
- **Long Segments:** Average segment length exceeds the context window of most Transformer architectures.
- **Structural Diversity:** Each municipality follows its own meeting template and phrasing conventions.

#### Advantages
- **Authentic Data:** Based on real-world municipal records rather than synthetic text.  
- **Bilingual Design:** Enables cross-lingual and translation-based evaluation (Portuguese ↔ English).  
- **Multi-Municipality Coverage:** Facilitates generalization studies using *Leave-One-Municipality-Out* validation.  
- **Fine-Grained Annotations:** Includes both segment-level (QA) and token-level (NER) labels for multi-task evaluation.

---

## 9. Architecture

### Component Descriptions

The MiNER framework is composed of two core components — a Question Answering (QA) model for segment detection and a Named Entity Recognition (NER) model for structured metadata extraction.

#### Stage 1 – Question Answering (QA)

- **Model:** [`deepset/xlm-roberta-large-squad2`](https://huggingface.co/deepset/xlm-roberta-large-squad2)  
- **Objective:** Identify the *opening* and *closing* segments of each document that contain relevant metadata.  
- **Training Data:** Automatically generated SQuAD v2-style dataset built from annotated municipal minutes.  
- **Evaluation Metrics:**  
  - **F1-score:** Measures token-level overlap between predicted and gold-standard answers.  
  - **Exact Match (EM):** Percentage of predictions that exactly match the gold reference span.

#### Stage 2 – Named Entity Recognition (NER)

- **Models:**  
  - 🇵🇹 [`neuralmind/bert-large-portuguese-cased`](https://huggingface.co/neuralmind/bert-large-portuguese-cased) (*BERTimbau-Large*)  
  - 🌍 [`xlm-roberta-large`](https://huggingface.co/xlm-roberta-large) (*XLM-RoBERTa-Large*)  
- **Objective:** Extract token-level metadata entities (e.g., date, location, meeting type, participants, begin/end times).  
- **Evaluation Metrics:**  
  - **Precision (P)** – ratio of correctly predicted entities to all predicted entities  
  - **Recall (R)** – ratio of correctly predicted entities to all true entities  
  - **F1-score (F1)** – harmonic mean of precision and recall, computed using the `seqeval` library.

---

## 10. Reporting Issues

Please report any issues or bugs through the GitHub repository issue tracker:  
[**Repository URL**]()

When reporting an issue, please include the following details:

- Python version  
- CUDA and PyTorch version  
- Complete error message and stack trace  
- Minimal reproducible example (if applicable)

Providing this information helps ensure faster and more accurate debugging.

---

## 11. License

This project is licensed under **CC-BY-ND 4.0 (Creative Commons Attribution–NoDerivatives 4.0 International)**.

You are free to:

- **Share:** Copy and redistribute the material in any medium or format  

Under the following terms:

- **Attribution:** You must give appropriate credit.  
- **No Derivatives:** If you remix, transform, or build upon the material, you may not distribute the modified version.

For details, see the `LICENSE` file.

### Dataset License

The **Council Metadata Corpus** is derived from public municipal meeting minutes and is provided strictly for **research purposes only**.  
Original documents remain the copyright of their respective municipal governments.

---

## 12. Resources

### Models

Pre-trained models fine-tuned within the MiNER framework are available on the **Hugging Face Model Hub**:

- [`anonymous13542/BERTimbau-large-metadata-council-pt`](https://huggingface.co/anonymous13542/BERTimbau-large-metadata-council-pt) – Portuguese NER model  
- [`anonymous13542/XLMR-large-metadata-council-en`](https://huggingface.co/anonymous13542/XLMR-large-metadata-council-en) – English NER model  
- [`anonymous13542/XLMR-large-qa-council-pt`](https://huggingface.co/anonymous13542/XLMR-large-qa-council-pt) – Portuguese QA model  
- [`anonymous13542/XLMR-large-qa-council-en`](https://huggingface.co/anonymous13542/XLMR-large-qa-council-en) – English QA model  

### External Resources

- **seqeval Library:** [https://github.com/chakki-works/seqeval](https://github.com/chakki-works/seqeval)  
- **INCEpTION Annotation Tool:** [https://inception-project.github.io/](https://inception-project.github.io/)  

---

## 13. Acknowledgments

- Municipal governments of M1–M6 for providing access to meeting minutes  
- The **INCEpTION Project** for the annotation platform  
- **Hugging Face** for hosting models and providing the Transformers library  
- The **seqeval** project for sequence evaluation metrics  

---

_Last updated: October 14, 2025_
