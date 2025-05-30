# data_loader.py
# Dependencies
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset
from config import config

pd.set_option('display.max_columns', None)

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

def get_attack_category(file_name, class_config):
    # Determine which category mapping to use
    categories_to_use = None
    if class_config == 2:
        categories_to_use = ATTACK_CATEGORIES_2
    elif class_config == 6:
        categories_to_use = ATTACK_CATEGORIES_6
    elif class_config == 19:
        categories_to_use = ATTACK_CATEGORIES_19
    else:
        raise ValueError(f"Invalid class_config: {class_config}. Choose 2, 6, or 19.")

    for key_in_filename_map in categories_to_use:
        if key_in_filename_map in file_name: 
            return categories_to_use[key_in_filename_map] 

    if 'Benign' in categories_to_use and 'Benign' in file_name:
        return categories_to_use['Benign']
        
    print(f"Warning: Could not determine category for file: {file_name} with class_config: {class_config}. Returning 'Unknown'.")
    return 'Unknown Category'
    
def textualize_flow(row, feature_names):
    """Coverting to text"""
    text_parts = []
    for feature_name in feature_names:
        if feature_name in row:
            value = row[feature_name]
            clean_feature_name = feature_name.replace(' ', '_').replace('/', '_').replace('.','_')
            if isinstance(value, float):
                text_parts.append(f"{clean_feature_name}_{value:.2f}")
            else:
                text_parts.append(f"{clean_feature_name}_{str(value)}")
    return " [SEP]".join(text_parts)
                         
def load_and_prepare_data(data_dir, class_config, tokenizer, max_seq_len, test_size_for_val=0.2, random_state=42):
    print(f"Loading and preparing datasets for {class_config}-class configuration...")
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

    train_df_full = pd.concat(df_list_train, ignore_index=True)
    test_df = pd.concat(df_list_test, ignore_index=True)

    train_df_full['Attack_Type_Str'] = train_df_full['filename'].apply(lambda x: get_attack_category(x, class_config))
    test_df['Attack_Type_Str'] = test_df['filename'].apply(lambda x: get_attack_category(x, class_config))

    # Drop rows where Attack_Type could not be determined
    train_df_full = train_df_full[train_df_full['Attack_Type_Str'] != 'Unknown_Category_From_Filename'].copy()
    test_df = test_df[test_df['Attack_Type_Str'] != 'Unknown_Category_From_Filename'].copy()
    
    if train_df_full.empty or test_df.empty:
        raise ValueError("No data remaining after filtering for unknown categories. Check filename and category mappings.")
    
    # Feature column definition
    potential_feature_cols = [col for col in train_df_full.columns if col not in ['filename', 'Attack_Type_Str']]

    # Textualize
    print("Textualizing data...")
    train_df_full.loc[:, 'text'] = train_df_full.apply(lambda row: textualize_flow(row, potential_feature_cols), axis=1)
    test_df.loc[:, 'text'] = test_df.apply(lambda row: textualize_flow(row, potential_feature_cols), axis=1)

    label_encoder = LabelEncoder()
    train_df_full.loc[:, 'label'] = label_encoder.fit_transform(train_df_full['Attack_Type_Str'])
    test_df.loc[:, 'label'] = label_encoder.fit_transform(test_df['Attack_Type_Str'])
    
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}, classes: {list(label_encoder.classes_)}")
    print(f"Class mapping: {dict(zip(label_encoder.classes_, range(num_classes)))}")

    # Split training data to create a validation set
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df_full['text'].tolist(), 
        train_df_full['label'].tolist(), 
        test_size=test_size_for_val, 
        random_state=random_state, 
        stratify=train_df_full['label'].tolist()
    )

    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_seq_len)
    
    train_ds = HFDataset.from_dict({'text': train_texts, 'label': train_labels}).map(tokenize_function, batched=True)
    val_ds = HFDataset.from_dict({'text': val_texts, 'label': val_labels}).map(tokenize_function, batched=True)
    test_ds = HFDataset.from_dict({'text': test_texts, 'label': test_labels}).map(tokenize_function, batched=True)

    # Calculating class weights
    og_train_labels_for_weights = train_df_full['label'].tolist()

    # Actual features
    actual_feature_names = potential_feature_cols

    # Return DataFrames for X, as textualization needs feature names and values
    return train_ds, val_ds, test_ds, label_encoder, og_train_labels_for_weights, actual_feature_names