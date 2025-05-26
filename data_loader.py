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

# Load Data
test_path = '/data/user/bsindala/PhD/Research/CICIoMT2024/WiFI and MQTT/attacks/CSV/test'
train_path = '/data/user/bsindala/PhD/Research/CICIoMT2024/WiFI and MQTT/attacks/CSV/train'

def load_and_concatenate(filepath):
    dataframes = []
    for filename in os.listdir(filepath):
        if filename.endswith('.csv'):
            file_path = os.path.join(filepath, filename)
            df = pd.read_csv(file_path)
            #df['source_file'] = filename
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)
    
test_df = load_and_concatenate(test_path)
train_df = load_and_concatenate(train_path)

print("Test Dataframe shape:", test_df.shape)
print("Train Dataframe shape:", train_df.shape)

print(test_df.head())

print()

print(train_df.head())

print()

# Missing Values
print(test_df.isnull().sum())
print(train_df.isnull().sum())

print()

print(train_df.info)

print()

print(train_df.dtypes)

# Remove Constants
# train_df = train_df.loc[:, (train_df != train_df.iloc[0]).any()]

# # Normalize the data using Min Max Scaling
# normalized_train_df = (train_df - train_df.min()) / (train_df.max() - train_df.min())

# # Bin each feature into 5 bins
# binned_train_df = normalized_train_df.apply(lambda x: pd.cut(x, bins=5, labels=False))

# # Generate token sequences
# tokens = []
# for index, row in binned_train_df.iterrows():
#     row_tokens = [f"T{index}_f{col}_bin{int(row[col])}" for col in binned_train_df.columns]
#     tokens.append(row_tokens)
    
# # Print token sequences
# for row_tokens in tokens:
#     print(row_tokens)

# --- Your ATTACK_CATEGORIES dictionaries and get_attack_category function ---
# (Keep these as you provided)
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
ATTACK_CATEGORIES_6 = { 'Spoofing': 'Spoofing',
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
    if class_config == 2:
        categories_to_use = ATTACK_CATEGORIES_2
    elif class_config == 6:
        categories_to_use = ATTACK_CATEGORIES_6
    else:  # Default to 19 classes
        categories_to_use = ATTACK_CATEGORIES_19

    # Now 'categories_to_use' is defined
    for key_in_filename_map in categories_to_use:
        if key_in_filename_map in file_name: # Check if the specific attack string key is in the filename
            return categories_to_use[key_in_filename_map] # Return the mapped general category

    # Fallback if no category key is found in the filename
    # This part needs to be robust. If a file is 'Benign.csv', it should be caught.
    # The ATTACK_CATEGORIES dictionaries should have an entry for 'Benign' if filenames reflect that.
    # For example, if ATTACK_CATEGORIES_19 = { ..., 'Benign': 'Benign', ... }
    # and a file is named 'Benign_flows.csv', then 'Benign' should be in 'Benign_flows.csv'.

    if 'Benign' in categories_to_use and 'Benign' in file_name: # Explicitly check for 'Benign'
        return categories_to_use['Benign']
        
    print(f"Warning: Could not determine category for file: {file_name} with class_config: {class_config}. Returning 'Unknown'.")
    return 'Unknown' # Fallback for files that don't match any key
    
# --- 1. Load and Preprocess Data (Adapted from your script) ---
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