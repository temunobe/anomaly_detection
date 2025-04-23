# Load Libraries and Dependencies
import os
import dpkt

from transformers import RobertaTokenizer, RobertaModel
from config import config

model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

print(output)

# Load Data
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
bluetooth_attack_train_path = config['bluetooth_attack_train']
bluetooth_attack_test_path = config['bluetooth_attack_test']
bluetooth_profiling_path = config['bluetooth_profiling']
wifi_mqtt_attack_path = config['wifi_mqtt_attack_test']
wifi_mqtt_attack_train_path = config['wifi_mqtt_attack_train']
wifi_mqtt_profiling_active_path = config['wifi_mqtt_profiling_active']
wifi_mqtt_profiling_broker_path = config['wifi_mqtt_profiling_broker']
wifi_mqtt_profiling_idle_path = config['wifi_mqtt_profiling_idle']
wifi_mqtt_profiling_interactions_path = config['wifi_mqtt_profiling_interactions']
wifi_mqtt_profiling_power_path = config['wifi_mqtt_profiling_power']

bluetooth_attack_train = load_files(bluetooth_attack_train_path)
bluetooth_attack_test = load_files(bluetooth_attack_test_path)
bluetooth_profiling = load_files(bluetooth_profiling_path)
wifi_mqtt_attack = load_files(wifi_mqtt_attack_path)
wifi_mqtt_attack_train = load_files(wifi_mqtt_attack_train_path)
wifi_mqtt_profiling_active = load_files(wifi_mqtt_profiling_active_path)
wifi_mqtt_profiling_broker = load_files(wifi_mqtt_profiling_broker_path)
wifi_mqtt_profiling_idle = load_files(wifi_mqtt_profiling_idle_path)
wifi_mqtt_profiling_interactions = load_files(wifi_mqtt_profiling_interactions_path)
wifi_mqtt_profiling_power = load_files(wifi_mqtt_profiling_power_path)

print(bluetooth_profiling)