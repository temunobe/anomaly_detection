# from tensorflow.keras.utils import to_categorical # Might not be needed if RoBERTa takes integer labels
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import numpy as np

# --- Configuration ---
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "./roberta_iomt_classifier_output"
LOGGING_DIR = "./roberta_iomt_logs"
NUM_EPOCHS = 3 # Adjust as needed
BATCH_SIZE = 16 # Adjust based on GPU memory
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 256 # Max length of textualized sequences
CLASS_CONFIG = 19 # Choose 19, 6, or 2 based on your experiment

# --- 2. Textualization Function ---
# (Using the feature list you provided earlier)
# This list should ideally be derived from the 'feature_columns' returned by load_iomt_data
# For now, let's assume 'feature_columns' from the loader is the definitive list.

def textualize_flow(row, feature_names):
    """Converts a row of flow data (a pandas Series) into a textual string."""
    # Option 1: "Feature is Value. Feature is Value." (McCormick/APT-LLM style)
    # text_parts = []
    # for feature_name in feature_names:
    #     if feature_name in row: # Check if feature exists in the row
    #         value = row[feature_name]
    #         text_parts.append(f"{feature_name.replace('_', ' ').replace(' ', '_')} is {value:.2f}" if isinstance(value, float) else f"{feature_name.replace('_', ' ').replace(' ', '_')} is {value}")
    # return ". ".join(text_parts) + "."

    # Option 2: NIDS-GPT inspired (simpler adaptation for flow stats)
    # "feat1_val1 feat1_val2... [SEP] feat2_val1..."
    # For flow features, direct value representation with feature name prefix might be better than digit-splitting every number.
    text_parts = []
    for feature_name in feature_names:
        if feature_name in row: # Check if feature exists in the row
            value = row[feature_name]
            # Sanitize feature name for use as a "token prefix"
            clean_feature_name = feature_name.replace(' ', '_').replace('/', '_').replace('.', '_') # Example sanitization
            if isinstance(value, float):
                # For floats, you might decide to format them, e.g., to 2 decimal places, or use their string representation
                # NIDS-GPT digit splitting: text_parts.append(f"{clean_feature_name}_{'_'.join(list(str(value).replace('.', 'dot')))}")
                text_parts.append(f"{clean_feature_name}_{value:.2f}") # Simpler: feature_value
            else:
                text_parts.append(f"{clean_feature_name}_{str(value)}")
    return " [SEP] ".join(text_parts)


# --- 3. PyTorch Dataset Class ---
class IoMTFlowDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 4. Evaluation Metrics ---
def compute_metrics_fn(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

