# data_loader.py
# Dependencies
import os
import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import RobertaTokenizer

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pd.set_option('display.max_columns', None)
    
def textualize_flow(row, feature_names, sep_token="</s>"):
    """Coverting flow data to concise text for RoBERTa, minimizing bloat."""
    text_parts = []
    for feature_name in feature_names:
        if feature_name in row:
            value = row[feature_name]
            # Simplify feature names to reduce token count
            clean_feature_name = feature_name.replace('_',' ').replace('-',' ').replace('/',' ')
            
            if pd.isnull(value):
                value = "missing"
            elif isinstance(value, float):
                value = f"{value:.2f}" if abs(value) >= 0.01 else f"{value:.4f}"
            elif isinstance(value, int):
                value = str(value)
            else:
                value = str(value)
                
            # Concise key-value format 
            text_parts.append(f"{clean_feature_name} is {value}")
            
    return f" {sep_token}".join(text_parts)
                         
def load_and_prepare_data(data_dir, tokenizer=None, max_seq_len=256, test_size_for_val=0.2, random_state=42, sample_frac=0.1, selected_features=None):
    """Load and prepare WUSTL-EHMS-2020 dataset for RoBERTa fine-tuning."""
    logger.info(f"Loading WUSTL-EHMS-2020 dataset for binary classification...")
    
    csv_path = os.path.join(data_dir, "wustl-ehms-2020_with_attacks_categories.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}.")

    df = pd.read_csv(csv_path)
    df = df.fillna(df.mean(numeric_only=True))
                   
    # Binary label
    df = df[df['Label'].isin([0, 1])]
    if df['Label'].nunique() != 2:
        raise ValueError(f"Expected 2 classes, found {df['Label'].nunique()} unique labels.")
    
    if selected_features is None:
        selected_features = [
            'SrcBytes', 'DstBytes', 'SrcPkts', 'DstPksts', 'TotPkts', 'TotBytes', 'Dur', 'Sport', 'Dport', # Network
            'ECG', 'BP', 'EE', 'HR', 'Sp02', 'GSR', 'TEMP', 'IBI' # Biometric
        ]
        
    # Filter valid features
    selected_features = [col for col in selected_features if col in df.columns]
    if not selected_features:
        raise ValueError("No valid features selected.")
            
    # Sampling data
    if sample_frac < 1.0:
        logger.info(f'Subsampling {sample_frac*100}% of data...')
        df = df.sample(frac=sample_frac, random_state=random_state)

    # Textualize for RoBERTa
    logger.info('Textualizing data...')
    df['text'] = df.apply(lambda row: textualize_flow(row, selected_features), axis=1)
    labels = df['Label'].values

    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'], 
        labels, 
        test_size=0.2, 
        random_state=random_state, 
        stratify=labels
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels, 
        test_size=test_size_for_val, 
        random_state=random_state,
        stratify=train_labels
    )
    
    logger.info(f'Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}, Test samples: {len(test_texts)}')

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_seq_len)
    
    train_ds = Dataset.from_dict({'text': train_texts.tolist(), 'label': train_labels}).map(tokenize_function, batched=True, batch_size=1000)
    val_ds = Dataset.from_dict({'text': val_texts.tolist(), 'label': val_labels}).map(tokenize_function, batched=True, batch_size=1000)
    test_ds = Dataset.from_dict({'text': test_texts.tolist(), 'label': test_labels}).map(tokenize_function, batched=True, batch_size=1000)
    
    # Calculating class weights
    try:
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = dict(enumerate(class_weights))
        logger.info(f"Class weights: {class_weights}")
    except Exception as e:
        logger.error(f"Failed to compute class weights: {e}")
        class_weights = {i: 1.0 for i in range(num_classes)}
        logger.info(f"Using equal class weights as fallback: {class_weights}")
        
    # Define id2label and label2id for RoBERTa    
    id2label = {0: "Benign", 1: "Attack"}
    label2id = {"Benign": 0, "Attack": 1}

    # Return DataFrames for X, as textualization needs feature names and values
    return train_ds, val_ds, test_ds, id2label, label2id, class_weights, selected_features