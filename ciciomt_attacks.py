#ciciomt_attacks
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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