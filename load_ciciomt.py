# load_ciciomt.py
# Dependencies
import os
import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset, concatenate_datasets
from ciciomt_attacks import get_attack_category

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pd.set_option('display.max_columns', None)
    
def textualize_flow(row, feature_names, sep_token="</s>") -> str:
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
            if 'bytes' in clean_feature_name.lower():
                text_parts.append(f"The {clean_feature_name} is {value} bytes")
            elif 'time' in clean_feature_name.lower() or 'duration' in clean_feature_name.lower():
                text_parts.append(f"The {clean_feature_name} is {value} seconds")
            else:
                text_parts.append(f"The {clean_feature_name} is {value}")
            
    return f" {sep_token}".join(text_parts)
                         
def load_and_prepare_data(data_dir, class_config, tokenizer=None, max_seq_len=256, test_size_for_val=0.2, random_state=42, sample_frac=0.2, feature_cols=None):#sample_size=None):
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
        df['text'] = df.apply(lambda row: textualize_flow(row, cols), axis=1)
        return df[['text', 'Attack_Type']]
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

    # Split training data to create a validation set
    train_df, val_df = train_test_split(
        train_df, 
        test_size=test_size_for_val, 
        random_state=random_state, 
        stratify=train_df['label']
    )
    
    logger.info(f'Training samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}')

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