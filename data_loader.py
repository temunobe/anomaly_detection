# data_loader.py
# Dependencies
import os
import glob
import torch
import requests
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel
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
        raise ValueError(f"Invalid class_config: {class_config}. Choose 2, 6, or 19."

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
                         
def load_iomt_data(data_dir, class_config, test_size_for_val=0.2):
    print(f"Loading data for {class_config}-class configuration...")
    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Ensure '{data_dir}' contains 'train' and 'test' subdirectories.")

    train_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.csv')]
    test_files = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.csv')]

    if not train_files or not test_files:
        raise FileNotFoundError("No CSV files found in train or test directories.")

    train_df_list = []
    for f in train_files:
        try:
            df_temp = pd.read_csv(f)
            df_temp['label_source_file'] = os.path.basename(f) # Keep original filename for label derivation
            train_df_list.append(df_temp)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
    
    test_df_list = []
    for f in test_files:
        try:
            df_temp = pd.read_csv(f)
            df_temp['label_source_file'] = os.path.basename(f)
            test_df_list.append(df_temp)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue

    if not train_df_list or not test_df_list:
        raise ValueError("Could not load any data from CSV files.")

    train_df_full = pd.concat(train_df_list, ignore_index=True)
    test_df = pd.concat(test_df_list, ignore_index=True)

    train_df_full['Attack_Type'] = train_df_full['label_source_file'].apply(lambda x: get_attack_category(x, class_config))
    test_df['Attack_Type'] = test_df['label_source_file'].apply(lambda x: get_attack_category(x, class_config))

    # Drop rows where Attack_Type could not be determined
    train_df_full = train_df_full[train_df_full['Attack_Type'] != 'Unknown_Attack_Type_In_Filename']
    test_df = test_df[test_df['Attack_Type'] != 'Unknown_Attack_Type_In_Filename']
    
    # Features (X) and labels (y)
    # We will keep the original features for textualization, so we don't drop them yet.
    # The 'label_source_file' is no longer needed after deriving 'Attack_Type'.
    
    y_train_full_str = train_df_full['Attack_Type']
    y_test_str = test_df['Attack_Type']

    label_encoder = LabelEncoder()
    y_train_full_encoded = label_encoder.fit_transform(y_train_full_str)
    y_test_encoded = label_encoder.transform(y_test_str)
    
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {dict(zip(label_encoder.classes_, range(num_classes)))}")

    # Create DataFrame for features that will be textualized
    # Exclude the label and source file columns from the features for textualization
    feature_columns = [col for col in train_df_full.columns if col not in ['Attack_Type', 'label_source_file']]
    
    X_train_full_df = train_df_full[feature_columns]
    X_test_df = test_df[feature_columns]

    # Split training data to create a validation set
    X_train_df, X_val_df, y_train_encoded, y_val_encoded = train_test_split(
        X_train_full_df, y_train_full_encoded, 
        test_size=test_size_for_val, 
        random_state=42, 
        stratify=y_train_full_encoded # Stratify for imbalanced datasets
    )
    
    print(f"Training samples: {len(X_train_df)}")
    print(f"Validation samples: {len(X_val_df)}")
    print(f"Test samples: {len(X_test_df)}")

    # Return DataFrames for X, as textualization needs feature names and values
    return X_train_df, X_val_df, X_test_df, y_train_encoded, y_val_encoded, y_test_encoded, label_encoder, feature_columns