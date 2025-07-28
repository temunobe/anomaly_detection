# data_loader.py
# Dependencies
import os
import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset as HFDataset
#from config import config

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pd.set_option('display.max_columns', None)

# 19 class mapping
ATTACK_CATEGORIES_19 = {
    'ARP_Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT-DDoS-Connect_Flood',
    'MQTT-DDoS-Publish_Flood': 'MQTT-DDoS-Publish_Flood',
    'MQTT-DoS-Connect_Flood': 'MQTT-DoS-Connect_Flood',
    'MQTT-DoS-Publish_Flood': 'MQTT-DoS-Publish_Flood',
    'MQTT-Malformed_Data': 'MQTT-Malformed_Data',
    'Recon-OS_Scan': 'Recon-OS_Scan',
    'Recon-Ping_Sweep': 'Recon-Ping_Sweep',
    'Recon-Port_Scan': 'Recon-Port_Scan',
    'Recon-VulScan': 'Recon-VulScan',
    'TCP_IP-DDoS-ICMP': 'DDoS-ICMP',
    'TCP_IP-DDoS-SYN': 'DDoS-SYN',
    'TCP_IP-DDoS-TCP': 'DDoS-TCP',
    'TCP_IP-DDoS-UDP': 'DDoS-UDP',
    'TCP_IP-DoS-ICMP': 'DoS-ICMP',
    'TCP_IP-DoS-SYN': 'DoS-SYN',
    'TCP_IP-DoS-TCP': 'DoS-TCP',
    'TCP_IP-DoS-UDP': 'DoS-UDP',
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
    # Determine which category mapping to use
    # categories_to_use = {
    #     2: ATTACK_CATEGORIES_2,
    #     6: ATTACK_CATEGORIES_6,
    #     19: ATTACK_CATEGORIES_19
    # }.get(class_config)
    
    # if not categories_to_use:
    #     raise ValueError(f"Invalid class_config: {class_config}. Choose 2, 6, or 19.")

    # category = categories_to_use.get(label, 'Unknown')
    # if category == 'Unknown':
    #     logger.warning(f"Could not map label: {label} with class_config: {class_config}. Returning 'Unknown'.")
    # return category
    if class_config == 2:
        categories = ATTACK_CATEGORIES_2
    elif class_config == 6:
        categories = ATTACK_CATEGORIES_6
    else:  # Default to 19 classes 
        categories = ATTACK_CATEGORIES_19  

    for key in categories:
        if key in label:
            return categories[key]
    
def textualize_flow(row, feature_names, sep_token="[SEP]"):
    """Coverting to text"""
    text_parts = []
    for feature_name in feature_names:
        if feature_name in row:
            value = row[feature_name]
            clean_feature_name = feature_name.replace(' ', '_').replace('/', '_').replace('.','_')
            
            if pd.isnull(value):
                value = "missing"
            elif isinstance(value, float):
                value = f"{value:.2f}" if abs(value) >= 0.01 else f"{value:.4f}"
            elif isinstance(value, int):
                value = str(value)
            else:
                value = str(value)
                
            if 'bytes' in clean_feature_name.lower():
                text_parts.append(f"The {clean_feature_name} is {value} bytes")
            elif 'time' in clean_feature_name.lower() or 'duration' in clean_feature_name.lower():
                text_parts.append(f"The {clean_feature_name} is {value} seconds")
            else:
                text_parts.append(f"The {clean_feature_name} is {value}")
            
    return f" {sep_token}".join(text_parts)
                         
def load_and_prepare_data(data_dir, class_config, tokenizer, max_seq_len, test_size_for_val=0.2, random_state=42, sample_size=None):
    logger.info(f"Loading and preparing datasets for {class_config}-class configuration...")
    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")

    if not os.path.exists(train_path) or not os.path.isdir(train_path):
        raise FileNotFoundError(f"Training directory not found or is not a directory: {train_path}.")
    if not os.path.exists(test_path) or not os.path.isdir(test_path):
        raise FileNotFoundError(f"Testing directory not found or is not a directory: {test_path}.")

    train_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.csv')]
    test_files = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.csv')]

    if not train_files:
        raise FileNotFoundError(f"No CSV files found in training directory: {train_path}")
    if not test_files:
        raise FileNotFoundError(f"No CSV files found in test directory: {test_path}.")

    df_list_train = [pd.read_csv(f).assign(filename=os.path.basename(f)) for f in train_files]
    df_list_test = [pd.read_csv(f).assign(filename=os.path.basename(f)) for f in test_files]

    train_df = pd.concat(df_list_train, ignore_index=True)
    test_df = pd.concat(df_list_test, ignore_index=True)
    
    # Handle missing values
    train_df = train_df.fillna(train_df.mean(numeric_only=True))
    test_df = test_df.fillna(test_df.mean(numeric_only=True))
    
    if sample_size:
        logger.info(f"Sampling {sample_size} instances from training data...")
        train_df = train_df.sample(n=sample_size, random_state=random_state)

    train_df['Attack_Type_Str'] = train_df['filename'].apply(lambda x: get_attack_category(x, class_config)) #filename for Label
    test_df['Attack_Type_Str'] = test_df['filename'].apply(lambda x: get_attack_category(x, class_config)) #filename for Label

    # Drop rows where Attack_Type could not be determined
    train_df = train_df[train_df['Attack_Type_Str'] != 'Unknown_Category_From_Filename'].copy()
    test_df = test_df[test_df['Attack_Type_Str'] != 'Unknown_Category_From_Filename'].copy()
    
    if train_df.empty or test_df.empty:
        raise ValueError("No data remaining after filtering for unknown categories. Check filename and category mappings.")
    
    # Feature column definition
    feature_cols = [col for col in train_df.columns if col not in ['filename', 'Attack_Type_Str']] #filename for Label

    # Textualize data
    logger.info("Textualizing data...")
    train_df['text'] = train_df.apply(lambda row: textualize_flow(row, feature_cols), axis=1)
    test_df['text'] = test_df.apply(lambda row: textualize_flow(row, feature_cols), axis=1)

    # Encoding labels
    all_labels = pd.concat([train_df['Attack_Type_Str'], test_df['Attack_Type_Str']]).unique()
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    train_df['label'] = label_encoder.transform(train_df['Attack_Type_Str'])
    test_df['label'] = label_encoder.transform(test_df['Attack_Type_Str'])
    
    num_classes = len(label_encoder.classes_)
    logger.info(f"Number of classes: {num_classes}, classes: {list(label_encoder.classes_)}")
    logger.info(f"Class mapping: {dict(zip(label_encoder.classes_, range(num_classes)))}")
    
    logger.info(f"Training samples (before split): {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    logger.info("Textualized Training Dataset\n", train_df.head())
    logger.info("Textualized Testing Dataset\n", test_df.head())

    # Split training data to create a validation set
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['text'].tolist(), 
        train_df['label'].tolist(), 
        test_size=test_size_for_val, 
        random_state=random_state, 
        stratify=train_df['label'].tolist()
    )

    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    logger.info(f"Test samples: {len(test_texts)}")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_seq_len)
    
    train_ds = HFDataset.from_dict({'text': train_texts, 'label': train_labels}).map(tokenize_function, batched=True)
    val_ds = HFDataset.from_dict({'text': val_texts, 'label': val_labels}).map(tokenize_function, batched=True)
    test_ds = HFDataset.from_dict({'text': test_texts, 'label': test_labels}).map(tokenize_function, batched=True)

    # Calculating class weights
    #og_train_labels_for_weights = train_df['label'].tolist()
    try:
        class_weights = compute_class_weight(
            class_weights='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = dict(enumerate(class_weights))
        logger.info(f"Computed class weights: {class_weights}")
    except Exception as e:
        logger.error(f"Failed to compute class weights: {e}")
        class_weights = {i: 1.0 for i in range(num_classes)}
        logger.info(f"Using equal class weights as fallback: m{class_weights}")

    # Actual features
    #actual_feature_names = feature_cols

    # Return DataFrames for X, as textualization needs feature names and values
    return train_ds, val_ds, test_ds, label_encoder, class_weights, feature_cols