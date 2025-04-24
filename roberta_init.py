# Load Libraries and Dependencies
import os
import glob
import torch
import requests
import pandas as pd

from PIL import Image
from sklearn.metrics import classification_report
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
    )
from config import config
from datasets import Dataset

model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

#print(output)

# Load Data
# Load CSV files
# Directories
wm_attack_test_dir = config['wm_attack_test']
wm_attack_train_dir = config['wm_attack_train']
wm_profiling_dir = config['wm_profiling']

wifi_mqtt_attack_test = glob.glob(os.path.join(wm_attack_test_dir, '*.csv'))
wifi_mqtt_attack_train = glob.glob(os.path.join(wm_attack_train_dir, '*.csv'))
wifi_mqtt_profiling = glob.glob(os.path.join(wm_profiling_dir, '*.csv'))

# Load and concatenate all CSV files
wm_test_df = [pd.read_csv(file) for file in wifi_mqtt_attack_test]
wifi_mqtt_test_df = pd.concat(wm_test_df, ignore_index=True)

wm_train_df = [pd.read_csv(file) for file in wifi_mqtt_attack_train]
wifi_mqtt_train_df = pd.concat(wm_train_df, ignore_index=True)

wm_profiling_df = [pd.read_csv(file) for file in wifi_mqtt_profiling]
wifi_mqtt_profiling_df = pd.concat(wm_profiling_df, ignore_index=True)

print(wifi_mqtt_test_df.columns)
print(wifi_mqtt_train_df.columns)
print(wifi_mqtt_profiling_df.columns)

# Preprocessing
# This function tokenizes the input text using the RoBERTa tokenizer. 
# It applies padding and truncation to ensure that all sequences have the same length (256 tokens).
def data_preprocess(data):
    data['input_ids'] = data['text'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length', max_length=128))
    data['attention_mask'] = data['input_ids'].apply(lambda x: [1 if token != tokenizer.pad_token_id else 0 for token in x])
    return data
    
# wm_test_data = data_preprocess(wifi_mqtt_test_df)
# wm_train_data = data_preprocess(wifi_mqtt_train_df)

# # Convert to hugging face dataset 
# wm_test_ds = Dataset.from_pandas(wm_test_data)
# wm_train_ds = Dataset.from_pandas(wm_train_data)

# print(wm_test_ds)

# Data files are in pcap format
def load_files(folder_path):
    """
    Load all PCAP files from a folder and return the packets.

    Parameters:
    folder_path (str): The path to the folder containing PCAP files.

    Returns:
    list: A list of packets from all PCAP files in the folder.
    """
    packets = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pcap'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        pcap = dpkt.pcap.Reader(f)
                        for timestamp, buf in pcap:
                            packets.append((timestamp, buf))
                        print(f"Loaded {file_path} successfully.")
                except Exception as e:
                    print(f"An error occurred while loading {file_path}: {e}")
    return packets

# Loading data paths
bluetooth_attack_train_path = config['bluetooth_train']
bluetooth_attack_test_path = config['bluetooth_test']
bluetooth_profiling_path = config['bluetooth_profiling']
wifi_mqtt_attack_test_path = config['wm_test']
wifi_mqtt_attack_train_path = config['wm_train']
wifi_mqtt_profiling_active_path = config['wm_active']
wifi_mqtt_profiling_broker_path = config['wm_broker']
wifi_mqtt_profiling_idle_path = config['wm_idle']
wm_interactions_bc = config['wm_interactions_bc']
wm_interactions_ec = config['wm_interactions_ec']
wm_interactions_mit = config['wm_interactions_mit']
wm_interactions_mc = config['wm_interactions_mc']
wm_interactions_ot = config['wm_interactions_ot']
wm_interactions_su = config['wm_interactions_su']
wm_interactions_sc = config['wm_interactions_sc']
wifi_mqtt_profiling_power_path = config['wm_power']

bluetooth_train = load_files(bluetooth_attack_train_path)
bluetooth_test = load_files(bluetooth_attack_test_path)
bluetooth_profiling = load_files(bluetooth_profiling_path)
wm_test = load_files(wifi_mqtt_attack_test_path)
wm_train = load_files(wifi_mqtt_attack_train_path)
wm_active = load_files(wifi_mqtt_profiling_active_path)
wm_broker = load_files(wifi_mqtt_profiling_broker_path)
wm_idle = load_files(wifi_mqtt_profiling_idle_path)
wm_interactions_bc = load_files(wm_interactions_bc)
wm_interactions_ec = load_files(wm_interactions_ec)
wm_interactions_mit = load_files(wm_interactions_mit)
wm_interactions_mc = load_files(wm_interactions_mc)
wm_interactions_ot = load_files(wm_interactions_ot)
wm_interactions_su = load_files(wm_interactions_su)
wm_interactions_sc = load_files(wm_interactions_sc)
wm_power = load_files(wifi_mqtt_profiling_power_path)

print(bluetooth_test)