import os
import pandas as pd
import numpy as np
import logging
import random

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pd.set_option('display.max_columns', None)

# Define feature groups

# CICIoMT2024 actual features
CIC_FEATURES = [
    'Header-Length', 'Protocol Type', 'Duration', 'Rate', 'Srate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP',
    'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
    'Radius', 'Covariance', 'Variance', 'Weight'
]

def descriptive_textualization(row, feature_names):
    """
    Dynamic generation of a descriptive sentence using all relevant features.
    """

    phrases = []
    for f in CIC_FEATURES:
        if f in feature_names and f in row and not pd.isnull(row[f]):
            v = row[f]
            phrases.append(f"{f}: {v}")
    # Label/Status
    label = row.get('Label', 0)
    status = "normal" if label == 0 else "anomaly"
    phrases.append(f"Status: {status}.")
    return ", ".join(phrases)

def textualize_flow(row, feature_names, sep_token="</s>", format_style="kv"):
    """Converts row data to text for RoBERTa input."""
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

def def load_and_prepare_data(
    data_dir, 
    class_config, 
    tokenizer=None, 
    max_seq_len=256, 
    test_size_for_val=0.2, 
    random_state=42, 
    sample_frac=0.2, 
    feature_cols=None, 
    normalize_numeric: bool = False, 
    oversample_minority: bool = False, 
    augment_numeric_jitter: float = 0.0):#sample_size=None):
    """Load and prepare CICIoMT2024 dataset efficiently with streaming and subsampling."""
    logger.info(f"Loading and preparing datasets for {class_config}-class configuration...")
    
    train_path, test_path = os.path.join(data_dir, "train"), os.path.join(data_dir, "test")

    if not os.path.exists(train_path) or not os.path.isdir(train_path):
        raise FileNotFoundError(f"Training directory not found or is not a directory: {train_path}.")
    if not os.path.exists(test_path) or not os.path.isdir(test_path):
        raise FileNotFoundError(f"Testing directory not found or is not a directory: {test_path}.")

    def process_file(file_path, class_config):
        df = pd.read_csv(file_path)
        df.fillna(df.mean(numeric_only=True), inplace=True)
        filename = os.path.basename(file_path)
        df['filename'] = filename
        df['Attack_Type'] = df['filename'].apply(lambda x: get_attack_category(x, class_config))
        df = df[df['Attack_Type'] != 'Unkown Category']
        
        # select features
        cols = feature_cols or [col for col in df.columns if col not in ['filename', 'Attack_Type']]
    # textualization will be applied after optional normalization/oversampling on splits
    df['Attack_Type'] = df['Attack_Type']
    return df
    # Using streaming to load data efficiently
    train_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.csv')]
    test_files = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.csv')]
    
    if not train_files or not test_files:
        raise FileNotFoundError(f"No CSV files found in training or test directories.")

    train_df = pd.concat([process_file(fp, class_config) for fp in train_files], ignore_index=True)
    test_df = pd.concat([process_file(fp, class_config) for fp in test_files], ignore_index=True)
            
    # Sampling train data
    if sample_frac < 1.0:
        logger.info(f'Subsampling {sample_frac*100}% of training data...')
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)

    # Encoding labels
    label_encoder = LabelEncoder()
    all_labels = pd.concat([train_df['Attack_Type'], test_df['Attack_Type']])
    label_encoder.fit(all_labels)
    train_df['label'] = label_encoder.transform(train_df['Attack_Type'])
    test_df['label'] = label_encoder.transform(test_df['Attack_Type'])
    
    num_classes = len(label_encoder.classes_)
    logger.info(f"Number of classes: {num_classes}, classes: {list(label_encoder.classes_)}")

    # Split training data to create a validation set (do this before textualization so we can normalize/augment)
    train_df, val_df = train_test_split(
        train_df,
        test_size=test_size_for_val,
        random_state=random_state,
        stratify=train_df['label']
    )

    logger.info(f'Samples after split -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}')

    # Determine feature columns to operate on
    cols = feature_cols or [col for col in train_df.columns if col not in ['filename', 'Attack_Type']]
    numeric_cols = [c for c in cols if c in train_df.columns and pd.api.types.is_numeric_dtype(train_df[c])]
    if numeric_cols:
        logger.info(f'Numeric columns detected for potential normalization: {numeric_cols}')

    # use explicit parameters passed to the function
    augment_jitter = augment_numeric_jitter

    # Normalize numeric columns using train stats
    if normalize_numeric and numeric_cols:
        stats = {}
        for c in numeric_cols:
            mean = train_df[c].astype(float).mean()
            std = train_df[c].astype(float).std()
            if pd.isna(std) or std == 0:
                std = 1.0
            stats[c] = (mean, std)
            train_df[c] = (train_df[c].astype(float) - mean) / std
            val_df[c] = (val_df[c].astype(float) - mean) / std
            test_df[c] = (test_df[c].astype(float) - mean) / std
        logger.info(f'Applied normalization for columns: {list(stats.keys())}')

    # Oversample minority class in training set if requested (simple resample with replacement)
    if oversample_minority:
        counts = train_df['label'].value_counts()
        max_count = counts.max()
        frames = [train_df]
        for cls, cnt in counts.items():
            if cnt < max_count:
                needed = max_count - cnt
                samples = train_df[train_df['label'] == cls].sample(n=needed, replace=True, random_state=random_state)
                frames.append(samples)
        train_df = pd.concat(frames).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        logger.info(f'Oversampled training set to balance classes; new counts: {train_df["label"].value_counts().to_dict()}')

    # Apply numeric jitter augmentation to training split only
    if augment_jitter and augment_jitter > 0.0 and numeric_cols:
        rng = np.random.default_rng(seed=random_state)
        col_stds = {}
        for c in numeric_cols:
            std = float(train_df[c].astype(float).std()) if pd.api.types.is_numeric_dtype(train_df[c]) else 1.0
            if std == 0 or pd.isna(std):
                std = 1.0
            col_stds[c] = std
        logger.info(f'Applying numeric jitter with factor {augment_jitter} to training set numeric columns')
        for c in numeric_cols:
            noise = rng.normal(loc=0.0, scale=augment_jitter * col_stds[c], size=len(train_df))
            train_df[c] = train_df[c].astype(float) + noise

    # Textualize after preprocessing
    train_df['text'] = train_df.apply(lambda row: textualize_flow(row, cols), axis=1)
    val_df['text'] = val_df.apply(lambda row: textualize_flow(row, cols), axis=1)
    test_df['text'] = test_df.apply(lambda row: textualize_flow(row, cols), axis=1)

    # Tokenize
    def tokenize_batch(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=max_seq_len)

    train_ds = Dataset.from_pandas(train_df[['text', 'label']]).map(tokenize_batch, batched=True)
    val_ds = Dataset.from_pandas(val_df[['text', 'label']]).map(tokenize_batch, batched=True)
    test_ds = Dataset.from_pandas(test_df[['text', 'label']]).map(tokenize_batch, batched=True)

    # Calculating class weights
    try:
        class_weights = compute_class_weight(
            #class_weight='balanced',
            'balanced',
            classes=np.unique(train_df['label']),
            y=train_df['label']
        )
        class_weights = dict(enumerate(class_weights))
        logger.info(f"Class weights: {class_weights}")
    except Exception as e:
        logger.error(f"Failed to compute class weights: {e}")
        class_weights = {i: 1.0 for i in range(num_classes)}
        logger.info(f"Using equal class weights as fallback: {class_weights}")

    # Actual features
    #actual_feature_names = feature_cols

    # Return DataFrames for X, as textualization needs feature names and values
    return train_ds, val_ds, test_ds, label_encoder, class_weights, feature_cols or train_df.columns.tolist()