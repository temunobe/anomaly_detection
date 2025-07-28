# main.py
import os
import numpy as np
import torch
import json
import logging

from transformers import RobertaTokenizerFast, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import from modules
from roberta_model import init_roberta_model, CustomTrainerWithWeightedLoss
from data_loader import load_and_prepare_data

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Data Directory
DATA_DIR = "/data/user/bsindala/PhD/Research/DataSets/CICIoMT2024/WiFI and MQTT/attacks/CSV/" 
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "/data/user/bsindala/PhD/Research/LLM_Anomaly_Detection/models"
LOGGING_DIR = "/data/user/bsindala/PhD/Research/LLM_Anomaly_Detection/logs"
NUM_EPOCHS = 3#10 #3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 128
CLASS_CONFIG = 19 # Choose 19, 6, or 2 based on your experiment
RANDOM_STATE = 42
SAVE_EVAL_RESULTS = True
SAMPLE_SIZE = None #10000 # For testing, None=Full Dataset

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
    
# Custom trainer
# class CustomTrainer(Trainer):
#     def __init__(self, class_weights, **kwargs):
#         super().__init__(**kwargs)
#         self.class_weights = class_weights
        
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits
#         loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
#         loss = loss_fct(logits, labels)
#         return (loss, outputs) if return_outputs else loss

# --- Main Execution ---
if __name__ == '__main__':
    try:
        # Initialization
        logger.info(f"Loading tokenizer for {MODEL_NAME}...")
        tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
        # Loading data
        logger.info(f"Loading and preprocessing data from {DATA_DIR}...")
        train_ds, val_ds, test_ds, label_encoder, class_weights, feature_names = load_and_prepare_data(
            data_dir=DATA_DIR,
            class_config=CLASS_CONFIG,
            tokenizer=tokenizer,
            max_seq_len=MAX_SEQ_LENGTH,
            random_state=RANDOM_STATE,
            sample_size=SAMPLE_SIZE
        )
        
        logger.info("Sample textualized data:")
        for i in range(min(3, len(train_ds))):
            logger.info(f"Text: {train_ds['text'][i]}")
            logger.info(f"Label: {label_encoder.inverse_transform([train_ds['label'][i]])[0]}")
    
        num_labels = len(label_encoder.classes_)
        id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
        label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
    
        logger.info(f"Number of unique labels: {num_labels}")
        logger.info(f"Training dataset size: {len(train_ds)}")
        logger.info(f"Validation dataset size: {len(val_ds)}")
        logger.info(f"Test dataset size: {len(test_ds)}")
        logger.info(f"Features used for textualization: {feature_names}")
        
        # Validate class
        if num_labels != {2:2, 6:6, 19:19}.get(CLASS_CONFIG):
            raise ValueError(f"Expected {CLASS_CONFIG} classes, but found {num_labels} classes.")
    
        # RoBERTa initialization
        logger.info(f"Initializing RoBERTa model {MODEL_NAME} for {num_labels} classes...")
        model = init_roberta_model(
            MODEL_NAME, 
            num_labels, 
            id2label=id2label, 
            label2id=label2id
        )
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Model loaded on: {device}")
    
        # Weight calculation and handling of imbalance
        logger.info("Using class weights from data_loader...")
        class_weights_tensor = torch.tensor(list(class_weigths.values()), dtype=torch.float).to(device)
        logger.info(f"Class weights: {class_weights_tensor.cpu().numpy().tolist()}")
        logger.info(f"Corresponding classes: {list(label_encoder.classes_)}")
    
        # Avoid division by 0
        # class_weights = 1. / np.where(class_counts == 0, 1, class_counts)
        # class_weights = class_weights / np.sum(class_weights) # Normalization
        # class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        
        # logger.info(f"Class weights: {class_weights_tensor.cpu().numpy().tolist()}")
        # logger.info(f"Corresponding classes: {list(label_encoder.classes_)}")
    
        # Training Arguments
        logger.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="linear",
            warmup_steps=500,
            warmup_ratio=0.1, # Warm up over 10% of training steps
            weight_decay=0.01,
            logging_dir=LOGGING_DIR,
            logging_steps=max(1, len(train_ds) // (BATCH_SIZE * 4)), # Log a few times per epoch
            logging_strategy="steps",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="tensorboard", # or "wandb", "mlflow", "none"
            fp16=torch.cuda.is_available(), # Enable mixed precision if cuda is available
            save_total_limit=2,
            seed=RANDOM_STATE,
            gradient_accumulation_steps=2,
            dataloader_num_workers=min(4, os.cpu_count() or 1), #os.cpu_count() // 2 if os.cpu_count() else 1 # for faster data loading
            disable_tqdm=False,
            gradient_checkpointing=True,
            torch_compile=False
        )
    
        # Init Trainer
        logger.info("Initializing Trainer...")
        trainer = CustomTrainerWithWeightedLoss(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            processing_class=data_collator,#tokenizer=tokenizer, # DataCollatorWithPadding will be used by default
            class_weights=class_weights_tensor # pass weights to custom trainer
        )
    
        # Model Training
        logger.info("Starting RoBERTa fine-tuning...")
        trainer.train()
    
        # Model evaluation on Test Set
        logger.info("\nEvaluating on the test dataset...")
        test_results = trainer.evaluate(eval_dataset=test_ds)
        logger.info(f"Test set evaluation results: {test_results}")
    
        # Save final model
        if SAVE_EVAL_RESULTS:
            eval_path = os.path.join(OUTPUT_DIR, "test_results.json")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(eval_path, "w") as f:
                json.dump(test_results, f, indent=4)
            logger.info(f"Test results saved to {eval_path}")
        
        # Save final model and tokenizer
        final_model_path = os.path.join(OUTPUT_DIR, "best_roberta_model") # load_best_model_at_end=True enables the trainer to save the best model
        model.save_pretrained(final_model_path) #os.path.join(OUTPUT_DIR, "final_roberta_epoch_model"))
        tokenizer.save_pretrained(final_model_path) #os.path.join(OUTPUT_DIR, "final_roberta_epoch_model_tokenizer"))
    
        # Saving label encoder classes for later use
        #label_encoder_path = os.path.join(final_model_path, "label_encoder_classes.npy")
        np.save(os.path.join(final_model_path, "label_encoder_classes.npy"), label_encoder.classes_)
        logger.info(f"Best model, tokenizer, and label encoder classes saved to {final_model_path}")
    
        # Baseline Model Training and Evaluation
        # Use original numerical features to train them.
    
        # For this, you'd need to re-load or pass X_train_df, y_train (integer encoded), etc.
        # from `load_and_prepare_datasets` before textualization if you want to use the raw features.
        # Or, modify `load_and_prepare_datasets` to also return the raw numerical dataframes.
    
        logger.info("\nScript execution completed successfully.")
    except Exception as e:
        logger.info(f"ERROR: An exception occured during execution: {e}")
        raise