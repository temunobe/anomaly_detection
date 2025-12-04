# run_roberta_ciciomt.py
import os
import numpy as np
import torch
import json
import logging
import wandb
import re

from transformers import RobertaTokenizerFast, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import from modules
from roberta_model import init_roberta_model, CustomTrainerWithWeightedLoss
from load_ciciomt import load_and_prepare_data
from config import config

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = 'true'
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
# Export HF token from config into environment so downstream HF calls can authenticate when loading private models.
try:
    _hf = config.get('hf_token') or config.get('HF_TOKEN')
    if _hf:
        os.environ.setdefault('HF_TOKEN', _hf)
        os.environ.setdefault('HUGGINGFACEHUB_API_TOKEN', _hf)
        logger.info('HF token exported to environment from config (for model downloads).')
except Exception:
    pass

# Data Directory
DATA_DIR = config['ciciomt_data_dir'] 
MODEL_NAME = "roberta-base"
OUTPUT_DIR = config['output']
LOGGING_DIR = config['logs'] 
WANDB_API_KEY = config['wb_api_key_ciciomt']
WANDB_PROJECT = config['wb_project_ciciomt']
WANDB_ENTITY = config['wb_entity']
NUM_EPOCHS = config['epochs']
BATCH_SIZE = config['batch_size']
LEARNING_RATE = config['lr']
MAX_SEQ_LENGTH = config['max_seq_len']
CLASS_CONFIG = config['class_config'] 
RANDOM_STATE = config['rand_state']
SAMPLE_FRAC = config['sample_frac'] 
SAVE_EVAL_RESULTS = True

# Metrics evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    metrics = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    if wandb.run is not None:
        wandb.log({"eval_" + k: v for k, v in metrics.items()})
    return metrics

# --- Main Execution ---
if __name__ == '__main__':
    try:
        # Intialization of WANDB
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            entity =WANDB_ENTITY,
            project=WANDB_PROJECT, 
            config={
                "model_name": MODEL_NAME,
                "num_epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "max_seq_length": MAX_SEQ_LENGTH,
                "class_config": CLASS_CONFIG,
                "sample_frac": SAMPLE_FRAC
            }
        )
        logger.info(f'W&B initialized for project: {WANDB_PROJECT}')
        
        # Initialization of tokenizer
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
            sample_frac=SAMPLE_FRAC,
            normalize_numeric=config.get('normalize_numeric', True),
            oversample_minority=config.get('oversample_minority', False),
            augment_numeric_jitter=config.get('augment_numeric_jitter', 0.0)
        )
        
        logger.info("Sample textualized data:")
        for i in range(min(3, len(train_ds))):
            logger.info(f"Text: {train_ds['text'][i]}")
            logger.info(f"Label: {label_encoder.inverse_transform([train_ds['label'][i]])[0]}")
        
        if wandb.run is not None:
            wandb.log({
                'training_size': len(train_ds),
                'val_size': len(val_ds),
                'test_size': len(test_ds),
                'num_classes': len(label_encoder.classes_),
                'features_used': len(feature_names)
            })
    
        num_labels = len(label_encoder.classes_)
        id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
        label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
    
        logger.info(f"Number of labels: {num_labels}, Classes: {list(label_encoder.classes_)}, Test size: {len(test_ds)}")
        if wandb.run is not None:
            wandb.log({'classes': list(label_encoder.classes_)})
        
        # Validate class config
        expected_classes = {2: 2, 6: 6, 19: 19}.get(CLASS_CONFIG)
        if num_labels != expected_classes:
            raise ValueError(f"Expected {expected_classes} classes, but found {num_labels}.")
    
        # RoBERTa initialization
        logger.info(f"Initializing RoBERTa model {MODEL_NAME} for {num_labels} classes...")
        model = init_roberta_model(
            model_name=MODEL_NAME, 
            num_labels=num_labels, 
            id2label=id2label, 
            label2id=label2id,
            use_lora=True,
            lora_r=16
        )
    
        # Decide precision (fp16 / bf16 / fp32) based on config and runtime support
        def decide_precision(cfg):
            pref = cfg.get('precision', 'auto')
            prefer_bf16 = cfg.get('prefer_bf16', False)
            cuda_available = torch.cuda.is_available()
            bf16_supported = False
            try:
                bf16_supported = cuda_available and getattr(torch.cuda, 'is_bf16_supported', lambda: False)()
            except Exception:
                bf16_supported = False

            fp16 = False
            bf16 = False
            if pref == 'fp32':
                fp16 = False
                bf16 = False
            elif pref == 'fp16':
                fp16 = cuda_available
                bf16 = False
            elif pref == 'bf16':
                if bf16_supported:
                    bf16 = True
                    fp16 = False
                else:
                    bf16 = False
                    fp16 = cuda_available
            else:  # auto
                if prefer_bf16 and bf16_supported:
                    bf16 = True
                    fp16 = False
                else:
                    fp16 = cuda_available
                    bf16 = False

            return fp16, bf16, bf16_supported, cuda_available

        logger.info("Setting up training arguments and precision...")
        fp16_flag, bf16_flag, bf16_supported, cuda_available = decide_precision(config)
        logger.info(f"Precision decision: fp16={fp16_flag}, bf16={bf16_flag}, bf16_supported={bf16_supported}, cuda_available={cuda_available}")

        if wandb.run is not None:
            wandb.config.update({
                'precision': config.get('precision', 'auto'),
                'prefer_bf16': config.get('prefer_bf16', False),
                'fp16': fp16_flag,
                'bf16': bf16_flag,
                'bf16_supported': bf16_supported
            })

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            gradient_accumulation_steps=4,
            fp16=fp16_flag,
            bf16=bf16_flag,
            gradient_checkpointing=True,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=50,
            report_to="none",
            max_grad_norm=1.0,
            dataloader_num_workers=0,
            save_total_limit=2,
            seed=RANDOM_STATE
        )
    
        # Init Trainer
        logger.info("Initializing Trainer...")
        trainer = CustomTrainerWithWeightedLoss(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            class_weights=class_weights,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        logger.info("Testing Dataloader for a single batch...")
        try:
            for i, batch in enumerate(trainer.get_train_dataloader()):
                logger.info(f"Batch {i} loaded: {batch.keys()}")
                if i > 0:
                    break
        except Exception as e:
            logger.error(f"Dataloader failed: {e}")
            raise
    
        # Model Training
        logger.info("Starting RoBERTa fine-tuning...")
        trainer.train()
    
        # Model evaluation on Test Set
        logger.info("\nEvaluating on the test dataset...")
        test_results = trainer.evaluate(eval_dataset=test_ds)
        logger.info(f"Test set evaluation results: {test_results}")
        if wandb.run is not None:
            wandb.log({'test_' + k: v for k, v in test_results.items()})
    
        # Save final model
        if SAVE_EVAL_RESULTS:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            eval_path = os.path.join(OUTPUT_DIR, "test_results_ciciomt.json")
            with open(eval_path, "w") as f:
                json.dump(test_results, f, indent=4)
            logger.info(f"Test results saved to {eval_path}")
        
        # Save final model and tokenizer
        final_model_path = os.path.join(OUTPUT_DIR, "best_roberta_model_ciciomt") 
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path) 
    
        # Saving label encoder classes for later use
        #label_encoder_path = os.path.join(final_model_path, "label_encoder_classes.npy")
        np.save(os.path.join(final_model_path, "label_encoder_classes.npy"), label_encoder.classes_)
        logger.info(f"Best model, tokenizer, and label encoder classes saved to {final_model_path}")
        
        if wandb.run is not None:
            artifact_name = os.path.basename(MODEL_NAME)
            artifact_name = re.sub(r'[^A-Za-z0-9_.\-]', '_', artifact_name)
            artifact = wandb.Artifact(artifact_name, type='model')
            artifact.add_dir(final_model_path)
            wandb.log_artifact(artifact)
            logger.info('Model and tokenizer saved as W&B artifact')
    
        logger.info("\nScript execution completed successfully.")
        wandb.finish()
    except Exception as e:
        emsg = str(e)
        if 'out of memory' in emsg.lower() or isinstance(e, RuntimeError) and 'out of memory' in emsg.lower():
            logger.error("CUDA out of memory detected. Suggestions:\n"
                         " - Reduce `batch_size` (per_device_train_batch_size) in config.py or training args.\n"
                         " - Close other GPU processes (check `nvidia-smi`), or run on a less-used GPU.\n"
                         " - Set environment variable PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 to reduce fragmentation.\n"
                         " - Use model sharding/device_map (if supported) or quantized/8-bit loading to lower memory.\n"
                         " - Disable other memory consumers or lower model size.")
        else:
            logger.info(f"ERROR: An exception occured during execution: {e}")

        if wandb.run is not None:
            wandb.log({'error': str(e)})
            wandb.finish()
        raise