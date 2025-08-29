# data_loader.py
# Dependencies
import os
import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from datasets import load_dataset, Dataset
#from config import config

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pd.set_option('display.max_columns', None)

# 19 class mapping
ATTACK_CATEGORIES_19 = {
    'ARP_Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT DDoS Connect Flood',
    'MQTT-DDoS-Publish_Flood': 'MQTT DDoS Publish Flood',
    'MQTT-DoS-Connect_Flood': 'MQTT DoS Connect Flood',
    'MQTT-DoS-Publish_Flood': 'MQTT DoS Publish Flood',
    'MQTT-Malformed_Data': 'MQTT Malformed Data',
    'Recon-OS_Scan': 'Recon OS Scan',
    'Recon-Ping_Sweep': 'Recon Ping Sweep',
    'Recon-Port_Scan': 'Recon Port Scan',
    'Recon-VulScan': 'Recon VulScan',
    'TCP_IP-DDoS-ICMP': 'DDoS ICMP',
    'TCP_IP-DDoS-SYN': 'DDoS SYN',
    'TCP_IP-DDoS-TCP': 'DDoS TCP',
    'TCP_IP-DDoS-UDP': 'DDoS UDP',
    'TCP_IP-DoS-ICMP': 'DoS ICMP',
    'TCP_IP-DoS-SYN': 'DoS SYN',
    'TCP_IP-DoS-TCP': 'DoS TCP',
    'TCP_IP-DoS-UDP': 'DoS UDP',
    'Benign': 'Benign'
}

# 6 Class mapping
ATTACK_CATEGORIES_6 = { 
    'Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT',
    'MQTT-DDoS-Publish_Flood': 'MQTT',
    'MQTT-DoS-Connect_Flood': 'MQTT',
    'MQTT-DoS-Publish_Flood': 'MQTT',
    'MQTT-Malformed_Data': 'MQTT',
    'Recon-OS_Scan': 'Recon',
    'Recon-Ping_Sweep': 'Recon',
    'Recon-Port_Scan': 'Recon',
    'Recon-VulScan': 'Recon',
    'DDoS-ICMP': 'DDoS',
    'DDoS-SYN': 'DDoS',
    'DDoS-TCP': 'DDoS',
    'DDoS-UDP': 'DDoS',
    'DoS-ICMP': 'DoS',
    'DoS-SYN': 'DoS',
    'DoS-TCP': 'DoS',
    'DoS-UDP': 'DoS',
    'Benign': 'Benign'
}

# 2 class mapping
ATTACK_CATEGORIES_2 = { #
    'ARP_Spoofing': 'attack',
    'MQTT-DDoS-Connect_Flood': 'attack',
    'MQTT-DDoS-Publish_Flood': 'attack',
    'MQTT-DoS-Connect_Flood': 'attack',
    'MQTT-DoS-Publish_Flood': 'attack',
    'MQTT-Malformed_Data': 'attack',
    'Recon-OS_Scan': 'attack',
    'Recon-Ping_Sweep': 'attack',
    'Recon-Port_Scan': 'attack',
    'Recon-VulScan': 'attack',
    'TCP_IP-DDoS-ICMP': 'attack',
    'TCP_IP-DDoS-SYN': 'attack',
    'TCP_IP-DDoS-TCP': 'attack',
    'TCP_IP-DDoS-UDP': 'attack',
    'TCP_IP-DoS-ICMP': 'attack',
    'TCP_IP-DoS-SYN': 'attack',
    'TCP_IP-DoS-TCP': 'attack',
    'TCP_IP-DoS-UDP': 'attack',
    'Benign': 'Benign'
}

def get_attack_category(label, class_config):
    """Map filename to attack category based on the class config."""
    if class_config == 2:
        categories = ATTACK_CATEGORIES_2
    elif class_config == 6:
        categories = ATTACK_CATEGORIES_6
    else:  # Default to 19 classes 
        categories = ATTACK_CATEGORIES_19  

    for key in categories:
        if key in label:
            return categories[key]
    logger.warning(f"Could not map label: {label} with class_config: {class_config}. Returning 'Unknown'.")
    return 'Unknown Category'
    
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
            text_parts.append(f"{clean_feature_name}:{value})
            
    return f" {sep_token}".join(text_parts)
                         
def load_and_prepare_data(data_dir, class_config, tokenizer, max_seq_len=256, test_size_for_val=0.2, random_state=42, simpla_frac=0.2):#sample_size=None):
    """Load and prepare CICIoMT2024 dataset efficiently with streaming and subsampling."""
    logger.info(f"Loading and preparing datasets for {class_config}-class configuration...")
    
    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")

    if not os.path.exists(train_path) or not os.path.isdir(train_path):
        raise FileNotFoundError(f"Training directory not found or is not a directory: {train_path}.")
    if not os.path.exists(test_path) or not os.path.isdir(test_path):
        raise FileNotFoundError(f"Testing directory not found or is not a directory: {test_path}.")

    # Using streaming to load data efficiently
    train_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.csv')]
    test_files = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.csv')]

    if not train_files or not test_files:
        raise FileNotFoundError(f"No CSV files found in training or test directories.")

    # Load with streaming=True to avoid memory overload
    logger.info("Loading datasets with streaming...")
    train_dataset = load_dataset('csv', data_files=train_files, streaming=True)
    test_dataset = load_dataset('csv', data_files=test_files, streaming=True)
    
    # Default to key features if not provided
    if selected_features is None:
        selected_features = [
            'Src IP', 'Dst IP', 'Protocol', 'Flow Duration', 'Pkt Len Mean',
            'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean', 'Flow Pkts/s', 'Flow IAT Mean',
            'Fwd IAT Tot', 'Bwd IAT Tot', 'Fwd PSH Flags', 'BWd PSH Flags'
        ]
    
    def process_batch(batch, feature_cols):
        """Process a batch of data for textualization and labeling"""
        df = pd.DataFrame(batch)
        df = df.fillna(df.mean(numeric_only=True))
        df['Attack_Type'] = df['filename'].apply(lambda x: get_attack_category(x, class_config))
        df = df[df['Attack_Type'] != 'Unknown Category'].copy()
        if df.empty:
            logger.warning('Empty batch after filtering unknown categories.')
            return None
        feature_cols = [col for col in df.columns if col not in ['filename', 'Attack_Type']]
        df['text'] = df.apply(lambda row: textualize_flow(row, feature_cols), axis=1)
        return df[['text', 'Attack_Type']]
        
    # Process train and test data
    feature_cols = selected_features
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    
    logger.info('Processing train data...')
    for batch in train_dataset['train']:
        df_batch = process_batch(batch, feature_cols)
        if df_batch is not None:
            # if feature_cols is None:
            #     feature_cols = [col for col in df_batch.columns if col not in ['text', 'Attack_Type', 'filename']]
            train_texts.extend(df_batch['text'].tolist())
            train_labels.extend(df_batch['Attack_Type'].tolist())
            
    logger.info('Processing test data...')
    for batch in test_dataset['train']:
        df_batch = process_batch(batch, feature_cols)
        if df_batch is not None:
            test_texts.extend(df_batch['text'].tolist())
            test_labels.extend(df_batch['Attack_Type'].tolist())
            
    # Sampling train data
    if sample_frac < 1.0:
        logger.info(f'Subsampling {sample_frac*100}% of training data...')
        indices = np.random.choice(len(train_texts), size=int(len(train_texts) * sample_frac), replace=False)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]

    # Encoding labels
    label_encoder = LabelEncoder()
    all_labels = list(set(train_labels + test_labels) #pd.concat([train_df['Attack_Type'], test_df['Attack_Type']]).unique()
    label_encoder.fit(all_labels)
    train_labels = label_encoder.transform(train_labels)
    test_labels = label_encoder.transform(test_labels)
    
    num_classes = len(label_encoder.classes_)
    logger.info(f"Number of classes: {num_classes}, classes: {list(label_encoder.classes_)}")

    # Split training data to create a validation set
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
    
    train_ds = Dataset.from_dict({'text': train_texts, 'label': train_labels}).map(tokenize_function, batched=True, batch_size=1000)
    val_ds = Dataset.from_dict({'text': val_texts, 'label': val_labels}).map(tokenize_function, batched=True, batch_size=1000)
    test_ds = Dataset.from_dict({'text': test_texts, 'label': test_labels}).map(tokenize_function, batched=True, batch_size=1000)

    # Calculating class weights
    try:
        class_weights = compute_class_weight(
            #class_weight='balanced',
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

    # Actual features
    #actual_feature_names = feature_cols

    # Return DataFrames for X, as textualization needs feature names and values
    return train_ds, val_ds, test_ds, label_encoder, class_weights, feature_cols