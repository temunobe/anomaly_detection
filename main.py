# main.py
import os
import numpy as np
import torch

from transformers import RobertaTokenizerFast, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import from modules
from roberta_model import init_roberta_model, CustomTrainerWithWeightedLoss
from data_loader import load_and_prepare_data

# Data Directory
DATA_DIR = "/data/user/bsindala/PhD/Research/CICIoMT2024/WiFI and MQTT/attacks/CSV/" 
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "/data/user/bsindala/PhD/Research/LLM_Anomaly_Detection/models"
LOGGING_DIR = "/data/user/bsindala/PhD/Research/LLM_Anomaly_Detection/logs"
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 236
CLASS_CONFIG = 19 # Choose 19, 6, or 2 based on your experiment
RANDOM_STATE = 42

# Metrics evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- Main Execution ---
if __name__ == '__main__':
    # Initialization
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

    # Loading data
    train_ds, val_ds, test_ds, label_encoder, og_train_labels, feature_names = load_and_prepare_data(
        data_dir=DATA_DIR,
        class_config=CLASS_CONFIG,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LENGTH,
        random_state=RANDOM_STATE
    )

    num_labels = len(label_encoder.classes_)
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in enumerate(label_encoder.classes_)}

    print(f"Number of unique labels: {num_labels}")
    print(f"Training dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    print(f"Features used for textualization: {feature_names}")

    # RoBERTa initialization
    print(f"Initializing RoBERTa model {MODEL_NAME} for {num_labels} classes...")
    model = init_roberta_model(MODEL_NAME, num_labels, id2label=id2label, label2id=label2id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on: {device}")

    # Weight calculation and handling of imbalance
    print("Calculating class weights for weighted loss...")
    class_counts = np.bincount(og_train_labels)

    # Avoid division by 0
    class_weights = 1. / np.where(class_counts == 0, 1, class_counts) # ** Replace 0 counts with 1 to avoid division by 0
    class_weights = class_weights / np.sum(class_weights) # Normalization
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    print(f"Class weights: {class_weights_tensor.cpu().numpy().tolist()}")
    print(f"Corresponding classes: {list(label_encoder.classes_)}")

    # Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1, # Warm up over 10% of training steps
        weight_decay=0.01,
        logging_dir=LOGGING_DIR,
        logging_steps=max(1, len(train_ds) // (BATCH_SIZE * 4)), # Log a few times per epoch
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="tensorboard", # or "wandb", "mlflow", "none"
        fp16=torch.cuda.is_available(), # Enable mixed precision if cuda is available
        save_total_limit=2,
        seed=RANDOM_STATE,
        dataloader_num_workers=os.cpu_count() // 2 if os.cpu_count() else 1 # for faster data loading
    )

    # Init Trainer
    print("Initializing Trainer...")
    trainer = CustomTrainerWithWeightedLoss(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer, # DataCollatorWithPadding will be used by default
        class_weights=class_weights_tensor # pass weights to custom trainer
    )

    # Model Training
    print("Starting RoBERTa fine-tuning...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        # TODO: potentially save some state or log more details
        raise

    # Model evaluation on Test Set
    print("\nEvaluating on the test dataset...")
    test_results = trainer.evaluate(eval_dataset=test_ds)
    print(f"Test set evaluation results: {test_results}")

    # Save final model
    final_model_path = os.path.join(OUTPUT_DIR, "best_roberta_model") # load_best_model_at_end=True enables the trainer to save the best model
    # Saving the very last state
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_roberta_epoch_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_roberta_epoch_model_tokenizer"))

    # Saving label encoder classes for later use
    label_encoder_path = os.path.join(final_model_path, "label_encoder_classes.npy")
    if not os.path.exists(final_model_path):
        os.makedirs(final_model_path) # if trainer didn't create it
    np.save(label_encoder_path, label_encoder.classes_)
    print(f"Best model, tokenizer, and label encoder classes saved to {final_model_path}")

    # Baseline Model Training and Evaluation
    # Use original numerical features to train them.

    # For this, you'd need to re-load or pass X_train_df, y_train (integer encoded), etc.
    # from `load_and_prepare_datasets` before textualization if you want to use the raw features.
    # Or, modify `load_and_prepare_datasets` to also return the raw numerical dataframes.

    print("\nScript execution finished.")