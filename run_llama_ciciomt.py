# run_llama_wustl.py
import os
import numpy as np
import torch
import json
import logging
import wandb
import re

from transformers import LlamaTokenizerFast, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report

# Import from modules
from llama_model import init_llama_model, CustomTrainerWithWeightedLoss
from data_loader import load_and_prepare_data
from config import config

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = 'true'
# Try to reduce CUDA memory fragmentation; helps avoid some OOMs
# See: https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
# Export HF token from config into environment so downstream HF calls can authenticate
try:
    _hf = config.get('hf_token') or config.get('HF_TOKEN')
    if _hf:
        os.environ.setdefault('HF_TOKEN', _hf)
        os.environ.setdefault('HUGGINGFACEHUB_API_TOKEN', _hf)
        logger.info('HF token exported to environment from config (for model downloads).')
except Exception:
    pass

# Data Directory and config
DATA_DIR = config['ciciomt_data_dir']
OUTPUT_DIR = config['output']
MODEL_NAME = config['llama_dir']
LOGGING_DIR = config['logs']
WANDB_API_KEY = config['wb_api_key_ciciomt']
WANDB_PROJECT = config['wb_project_ciciomt']
WANDB_ENTITY = config['wb_entity']
NUM_EPOCHS = config['epochs']
BATCH_SIZE = config['batch_size']
LEARNING_RATE = config['lr']
MAX_SEQ_LENGTH = config['max_seq_len']
RANDOM_STATE = config['rand_state']
SAMPLE_FRAC = config['sample_frac']
SAVE_EVAL_RESULTS = True
USE_FOCAL_LOSS = config['use_focal_loss']
CLASS_CONFIG = config.get('ciciomt_class_config', 2)  # 2, 6, or 19
PADDING_STRATEGY = config['padding_strategy']
FORMAT_STYLE = config['format_style']

# Metrics evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, pred.predictions[:,1]) if pred.predictions.shape[1] == 2 else 0.0
    except:
        auc = 0.0
    metrics = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }
    if wandb.run is not None:
        wandb.log({"eval_" + k: v for k, v in metrics.items()})
    return metrics

# --- Main Execution ---
if __name__ == '__main__':
    try:
        # Intialization of WANDB
        logger.info("Logging into Weights & Biases...")
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
                "sample_frac": SAMPLE_FRAC,
                "use_focal_loss": USE_FOCAL_LOSS,
                "format_style": FORMAT_STYLE,
                "padding_strategy": PADDING_STRATEGY,
                "kfold": KFOLD
            }
        )
        logger.info(f'W&B initialized for project: {WANDB_PROJECT}')
        
        # Initialization of tokenizer (robust: try fast, fall back to slow; ensure pad_token exists)
        logger.info(f"Loading tokenizer for {MODEL_NAME}...")
        try:
            tokenizer = LlamaTokenizerFast.from_pretrained(MODEL_NAME, local_files_only=True)
        except Exception as e_fast:
            logger.warning(f"Fast tokenizer load failed ({e_fast}), retrying with slow tokenizer...")
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        if tokenizer.pad_token is None:
            if getattr(tokenizer, 'eos_token', None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info('Set tokenizer.pad_token = tokenizer.eos_token')
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.info('Added special token [PAD] as tokenizer.pad_token')
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
        # Loading data (CICIoMT2024)
        logger.info(f"Loading and preprocessing CICIoMT2024 data from {DATA_DIR} (all CSVs in train/ and test/ subdirectories)...")
        train_ds, val_ds, test_ds, id2label, label2id, class_weights, feature_names = load_and_prepare_data(
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
        logger.info(f"Class Weights: {class_weights}")
        
        logger.info("Sample textualized data:")
        for i in range(min(3, len(train_ds))):
            logger.info(f"Text: {train_ds['text'][i]}")
            logger.info(f"Label: {id2label[train_ds['label'][i]]}")
        
        if wandb.run is not None:
            wandb.log({
                'training_size': len(train_ds),
                'val_size': len(val_ds),
                'test_size': len(test_ds),
                'num_classes': len(id2label),
                'features_used': len(feature_names)
            })
    
        num_labels = len(id2label)
    
        logger.info(f"Number of labels: {num_labels}, Classes: {list(id2label.values())}, Test size: {len(test_ds)}")
        if wandb.run is not None:
            wandb.log({'classes': list(id2label.values())})
    
        # Llama initialization
        logger.info(f"Initializing Llama model {MODEL_NAME} for {num_labels} classes...")
        model = init_llama_model(
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
                # Older torch may not expose is_bf16_supported; keep False
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
                    # fall back to fp16 when possible
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

        # expose chosen precision to W&B config
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
            lr_scheduler_type="cosine",
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
            use_focal_loss=USE_FOCAL_LOSS,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    
        # Model Training
        logger.info("Starting Llama fine-tuning...")
        trainer.train()
    
        # Model evaluation on Test Set
        logger.info("\nEvaluating on the test dataset...")
        test_results = trainer.evaluate(eval_dataset=test_ds)
        logger.info(f"Test set evaluation results: {test_results}")
        
        preds = trainer.predict(test_ds)
        pred_labels = preds.predictions.argmax(-1)
        true_labels = [int(x) for x in test_ds["label"]]
        cm = confusion_matrix(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels, target_names=list(id2label.values()))
        logger.info(f"Confusion matrix:\n{cm}")
        logger.info(f"Classification report:\n{report}")
        
        if wandb.run is not None:
            wandb.log({'test_' + k: v for k, v in test_results.items()})
            wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
                y_true=true_labels,
                preds=pred_labels,
                class_names=list(id2label.values())
            )})
                
        # Save final model
        if SAVE_EVAL_RESULTS:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            eval_path = os.path.join(OUTPUT_DIR, "test_results.json")
            with open(eval_path, "w") as f:
                json.dump(test_results, f, indent=4)
            with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
                f.write(report)
            logger.info(f"Test results and classification report saved to {eval_path}")
        
        # Save final model and tokenizer
        final_model_path = os.path.join(OUTPUT_DIR, "best_llama_model_v1") 
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path) 
    
        # Saving label encoder classes for later use
        #label_encoder_path = os.path.join(final_model_path, "label_encoder_classes.npy")
        #np.save(os.path.join(final_model_path, "label_encoder_classes.npy"), label_encoder.classes_)
        logger.info(f"Best model, tokenizer, and label encoder classes saved to {final_model_path}")
        
        if wandb.run is not None:
            # WandB artifact name must be a short alphanumeric-ish token.
            # MODEL_NAME may be a filesystem path; sanitize to a safe artifact name.
            artifact_name = os.path.basename(MODEL_NAME)
            artifact_name = re.sub(r'[^A-Za-z0-9_.\-]', '_', artifact_name)
            artifact = wandb.Artifact(artifact_name, type='model')
            artifact.add_dir(final_model_path)
            wandb.log_artifact(artifact)
            logger.info('Model and tokenizer saved as W&B artifact')
    
        logger.info("\nScript execution completed successfully.")
        wandb.finish()
    except Exception as e:
        # Specific handling for CUDA out-of-memory to give actionable advice
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