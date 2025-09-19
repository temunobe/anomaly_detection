# Anomaly Detection with RoBERTA, Llama, and Mistral

This is a unified framework for sequence-based anomaly detection on tabular and time-series datasets using large language models (LLMs) and transformer-based architectures. It supports **RoBERTa**, **Llama**, and **Mistral** models, leveraging both standard and parameter-efficient (LoRA) fine-tuning with quantization for memory efficiency.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
  - [RoBERTa](#roberta)
  - [Llama](#llama)
  - [Mistral](#mistral)
- [Data Processing & Textualization](#data-processing-textualization)
- [Training & Evaluation](#training-evaluation)
- [How to Run](#how-to-run)
- [Results](#results)
- [References](#references)

---

## Project Overview

The goal of this project is to detect anomalies in complex, heterogeneous data (including physiological, network, and device-context features) using state-of-the-art transformer models. The repository includes:
- Custom data loaders and textualizers for tabular/temporal features
- Model initialization and fine-tuning scripts for RoBERTa, Llama, and Mistral
- Support for LoRA adapters, 4/8-bit quantization (BitsAndBytes), class-weighted and focal loss
- Robust train/validation/test splits with KFold cross-validation
- Extensive logging and experiment tracking (Weights & Biases)

---

## Dataset

The main dataset used is **WUSTL-EHMS-2020**, a binary classification dataset with both normal and attack (anomaly) labels. The feature set includes physiological signals (e.g., HR, SpO2, EEG), network statistics (e.g., packet counts), and contextual metadata (e.g., time, device ID, location).

---

## Model Architectures

### RoBERTa

- **Script:** `run_roberta.py`
- **Model:** HuggingFace `roberta-base` (or custom checkpoint)
- **Usage:** Sequence classification head for anomaly detection
- **Features:** 
  - Supports class weighting and focal loss
  - Handles tabular-to-text conversion via descriptive or key-value textualization
  - Standard training arguments and evaluation metrics
  - Advanced logging, model saving, and W&B integration
  
### Llama

- **Script:** `run_llama_wustl.py`
- **Model:** Meta's Llama-3.1/70B (or compatible checkpoint)
- **Usage:** LlamaForSequenceClassification with LoRA Support
- **Features:** 
  - Parameter-efficient training (`q_proj` and `v_proj` targeted for LoRA)
  - 4-bit/8-bit quantization for efficient training
  - LoRA adapters for parameter-efficient fine-tuning
  - Configurable model/device setup
  - Handles large sequences with optimized memory usage
  - Advanced logging, model saving, and W&B integration

### Mistral

- **Script:** `run_mistral_wustl.py`
- **Model:** `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (or other Mistral checkpoints)
- **Usage:** AutoModelForSequenceClassification with LoRA and quantization
- **Features:**
  - Parameter-efficient training (`q_proj` and `v_proj` targeted for LoRA)
  - BitsAndBytesConfig for 4/8-bit quantization.
  - Advanced logging, model saving, and W&B integration

---

## Data Processing & Textualization

- Features are dynamically converted into natural language descriptions or key-value pairs.
- Example: `"HR is 80 bpm, SpO2 is 95%, Status: anomaly."`
- `mistral_load_wustl.py` and similar scripts handle loading, preprocessing, sampling, and textualization.

---

## Training & Evaluation

- **Class imbalance:** Supports class-weighted loss and focal loss.
- **Cross-validation:** Stratified KFold and random splits.
- **Metrics:** Accuracy, F1, precision, recall, AUC, confusion matrix, and classification report.
- **W&B:** Logs metrics, artifacts, and model checkpoints.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
- Place the `wustl-ehms-2020_with_attacks_categories.csv` in the specified data directory.

### 3. Run Experiments

#### RoBERTa
```bash
python run_roberta.py
```

#### Llama
```bash
python run_llama_wustl.py
```

#### Mistral
```bash
python run_mistral_wustl.py
```

*Configure model paths, batch size, epochs, etc. via `config.py`.*

---

## Results

- All models are evaluated and benchmarked on the same splits.
- Results and logs are saved to the output and logs directories, and optionally to Weights & Biases.

---

## References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Mistral AI](https://mistral.ai/)
- [Meta Llama](https://ai.meta.com/llama/)
- [WUSTL-EHMS-2020 Dataset](https://doi.org/10.1109/ACCESS.2020.2979856)

---

**Contact:** temunobe  
**License:** MIT