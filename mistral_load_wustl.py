#mistral_load_wustl.py

import os
import pandas as pd
import numpy as np
import logging
import random

from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pd.set_option('display.max_columns', None)

# Define feature groups
PHYSIO_FEATURES = ['HR', 'SpO2', 'TEMP', 'BP', 'ECG', 'EEG', 'IBI', 'GSR']
NETWORK_FEATURES = ['SrcBytes', 'DstBytes', 'TotBytes', 'SrcPkts', 'DstPkts', 'TotPkts', 'Dur', 'Sport', 'Dport']
DEVICE_CONTEXT_FEATURES = ['Time', 'DeviceID', 'Location', 'PatientID']

def descriptive_textualization(row, feature_names):
    """
    Dynamic generation of a descriptive sentence using all relevant features.
    """
    phrases = []
    # Contextual features
    if 'Time' in feature_names and 'Time' in row and not pd.isnull(row['Time']):
        phrases.append(f"At {row['Time']}")
    if 'DeviceID' in feature_names and 'DeviceID' in row and not pd.isnull(row['DeviceID']):
        phrases.append(f"device ID {row['DeviceID']}")
    if 'Location' in feature_names and 'Location' in row and not pd.isnull(row['Location']):
        phrases.append(f"at location {row['Location']}")
    if 'PatientID' in feature_names and 'PatientID' in row and not pd.isnull(row['PatientID']):
        phrases.append(f"for patient {row['PatientID']}")
        
    # Physiological features
    bio_phrases = []
    for f in PHYSIO_FEATURES:
        if f in feature_names and f in row and not pd.isnull(row[f]):
            v = row[f]
            unit = ""
            if f == 'HR':
                unit = " bpm"
            elif f == 'TEMP':
                unit = "°F"
            elif f == 'SpO2':
                unit = "%"
            bio_phrases.append(f"{f} of {v}{unit}")
    if bio_phrases:
        phrases.append(", ".join(bio_phrases))
        
    # Network features
    net_phrases = []
    for f in NETWORK_FEATURES:
        if f in feature_names and f in row and not pd.isnull(row[f]):
            v = row[f]
            net_phrases.append(f"{f}={v}")
    if net_phrases:
        phrases.append(", ".join(net_phrases))
        
    # Label/Status
    label = row.get('Label', 0)
    status = "normal" if label == 0 else "anomaly"
    phrases.append(f"Status: {status}.")
    return ", ".join(phrases)

def textualize_flow(row, feature_names, sep_token="</s>", format_style="kv"):
    """Converts row data to text for Mistral input."""
    if format_style == "descriptive":
        return descriptive_textualization(row, feature_names)
    
    text_parts = []
    for feature_name in feature_names:
        if feature_name in row:
            value = row[feature_name]
            clean_feature_name = feature_name.replace('_', ' ').replace('-', ' ').replace('/', ' ')
            if pd.isnull(value):
                value = "missing"
            elif isinstance(value, float):
                value = f"{value:.2f}" if abs(value) >= 0.01 else f"{value:.4f}"
            elif isinstance(value, int):
                value = str(value)
            else:
                value = str(value)
            if format_style == "kv":
                text_parts.append(f"{clean_feature_name} is {value}")
            elif format_style == "compact":
                text_parts.append(f"{clean_feature_name}:{value}")
            else:
                text_parts.append(f"{clean_feature_name} is {value}")
    return f" {sep_token}".join(text_parts)

def load_and_prepare_data(
    model,
    data_dir, 
    tokenizer=None, 
    max_seq_len=256, 
    test_size_for_val=0.2, 
    random_state=42, 
    sample_frac=0.1, 
    selected_features=None,
    format_style="kv",
    padding_strategy="max_length",
    kfold=None,
    normalize_numeric: bool = False,
    oversample_minority: bool = False,
    augment_numeric_jitter: float = 0.0,
):
    """Load and prepare WUSTL-EHMS-2020 dataset for Mistral fine-tuning."""
    logger.info(f"Loading WUSTL-EHMS-2020 dataset for binary classification...")
    
    csv_path = os.path.join(data_dir, "wustl-ehms-2020_with_attacks_categories.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}.")

    df = pd.read_csv(csv_path)
    # Safety: fill NA for all columns, including non-numeric
    df = df.fillna("missing")
    
    # Filter only binary labels
    df = df[df['Label'].isin([0, 1])]
    if df['Label'].nunique() != 2:
        raise ValueError(f"Expected 2 classes, found {df['Label'].nunique()} unique labels.")
    
    if selected_features is None:
        selected_features = PHYSIO_FEATURES + NETWORK_FEATURES + DEVICE_CONTEXT_FEATURES
    selected_features = [col for col in selected_features if col in df.columns]
    if not selected_features:
        raise ValueError("No valid features selected.")
    
    # Optional: Remove rows where all selected features are missing
    df = df.dropna(subset=selected_features, how="all")
        
    # Sampling data
    if sample_frac < 1.0:
        logger.info(f'Subsampling {sample_frac*100}% of data...')
        df = df.sample(frac=sample_frac, random_state=random_state)

    # Split into train/val/test (or kfold)
    labels = df['Label'].values
    if kfold is not None and kfold > 1:
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)
        for train_idx, test_idx in skf.split(df.index, labels):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            break
        train_df, val_df = train_test_split(
            train_df,
            test_size=test_size_for_val,
            random_state=random_state,
            stratify=train_df['Label']
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=random_state,
            stratify=df['Label']
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=test_size_for_val,
            random_state=random_state,
            stratify=train_df['Label']
        )
    logger.info(f'Samples after split -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}')

    # Optional numeric normalization (compute stats on train and apply to all splits)
    numeric_cols = []
    for col in selected_features:
        if col in train_df.columns and pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_cols.append(col)
    if numeric_cols:
        logger.info(f'Numeric columns detected for potential normalization: {numeric_cols}')

    # use explicit parameters passed to the function
    augment_jitter = augment_numeric_jitter

    # Normalize numeric columns using train stats
    if normalize_numeric and numeric_cols:
        stats = {}
        for c in numeric_cols:
            mean = train_df[c].replace('missing', np.nan).astype(float).mean()
            std = train_df[c].replace('missing', np.nan).astype(float).std()
            if pd.isna(std) or std == 0:
                std = 1.0
            stats[c] = (mean, std)
            train_df[c] = (train_df[c].replace('missing', np.nan).astype(float) - mean) / std
            val_df[c] = (val_df[c].replace('missing', np.nan).astype(float) - mean) / std
            test_df[c] = (test_df[c].replace('missing', np.nan).astype(float) - mean) / std
        logger.info(f'Applied normalization for columns: {list(stats.keys())}')

    # Oversample minority class in training set if requested (simple resample with replacement)
    if oversample_minority:
        counts = train_df['Label'].value_counts()
        max_count = counts.max()
        frames = [train_df]
        for cls, cnt in counts.items():
            if cnt < max_count:
                needed = max_count - cnt
                samples = train_df[train_df['Label'] == cls].sample(n=needed, replace=True, random_state=random_state)
                frames.append(samples)
        train_df = pd.concat(frames).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        logger.info(f'Oversampled training set to balance classes; new counts: {train_df["Label"].value_counts().to_dict()}')

    # Apply numeric jitter augmentation to training split only (adds Gaussian noise proportional to column std)
    if augment_jitter and augment_jitter > 0.0 and numeric_cols:
        rng = np.random.default_rng(seed=random_state)
        # compute stds from train_df (after normalization if applied, std may be 1.0)
        col_stds = {}
        for c in numeric_cols:
            # use original std if normalized (std==1.0), else compute current std
            std = float(train_df[c].astype(float).std()) if pd.api.types.is_numeric_dtype(train_df[c]) else 1.0
            if std == 0 or pd.isna(std):
                std = 1.0
            col_stds[c] = std
        logger.info(f'Applying numeric jitter with factor {augment_jitter} to training set numeric columns')
        for c in numeric_cols:
            noise = rng.normal(loc=0.0, scale=augment_jitter * col_stds[c], size=len(train_df))
            train_df[c] = train_df[c].astype(float) + noise

    # Textualize splits (after normalization/augmentation/oversampling)
    logger.info(f'Textualizing data using format: {format_style}')
    train_df['text'] = train_df.apply(lambda row: textualize_flow(row, selected_features, format_style=format_style), axis=1)
    val_df['text'] = val_df.apply(lambda row: textualize_flow(row, selected_features, format_style=format_style), axis=1)
    test_df['text'] = test_df.apply(lambda row: textualize_flow(row, selected_features, format_style=format_style), axis=1)

    train_texts = train_df['text'].tolist()
    val_texts = val_df['text'].tolist()
    test_texts = test_df['text'].tolist()
    train_labels = train_df['Label'].values
    val_labels = val_df['Label'].values
    test_labels = test_df['Label'].values

    tokenizer = AutoTokenizer.from_pretrained(model)

    # Apply preprocessing flags (explicit parameters)
    augment_jitter = augment_numeric_jitter

    # Tokenize
    def tokenize_function(examples):
        prompt = examples['text']#[f"[INST] {x} [/INST]" for x in examples['text']]
        return tokenizer(
            prompt, 
            padding=padding_strategy, 
            truncation=True, 
            max_length=max_seq_len
        )
    
    train_ds = Dataset.from_dict({'text': train_texts, 'label': train_labels}).map(tokenize_function, batched=True)
    val_ds = Dataset.from_dict({'text': val_texts, 'label': val_labels}).map(tokenize_function, batched=True)
    test_ds = Dataset.from_dict({'text': test_texts, 'label': test_labels}).map(tokenize_function, batched=True)

    # Check for NaN/Inf/extreme values in labels
    assert np.isfinite(np.array(train_labels)).all(), "Found Inf/NaN in train labels"
    assert np.isfinite(np.array(val_labels)).all(), "Found Inf/NaN in val labels"
    assert np.isfinite(np.array(test_labels)).all(), "Found Inf/NaN in test labels"
    
    # Calculating class weights
    try:
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = dict(enumerate([float(x) for x in class_weights]))
        logger.info(f"Class weights: {class_weights}")
    except Exception as e:
        logger.error(f"Failed to compute class weights: {e}")
        class_weights = {i: 1.0 for i in range(2)}
        logger.info(f"Using equal class weights as fallback: {class_weights}")
        
    id2label = {0: "Benign", 1: "Attack"}
    label2id = {"Benign": 0, "Attack": 1}

    return train_ds, val_ds, test_ds, id2label, label2id, class_weights, selected_features