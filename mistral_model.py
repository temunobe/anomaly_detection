# mistral_model.py

import torch
import logging
import wandb

from transformers import AutoModel, AutoTokenizer, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets):
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) **  self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal
        
def init_mistral_model(
    model_name,
    num_labels=2,
    id2label=None,
    label2id=None,
    use_lora=True,
    lora_r=16
):
    """
    Initializing Mistral model for sequence classifitication
    """
    logging.info(f"Initializing Mistral model: {model_name} with {num_labels} labels, LoRA={use_lora}")
    try:
        # Try to enable bitsandbytes 4-bit quantization if available. If bitsandbytes
        # is not installed, fall back to loading without quantization.
        quant_kwargs = {}
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            quant_kwargs['quantization_config'] = quantization_config
            logger.info("BitsAndBytesConfig created: attempting 4-bit quantized load.")
        except Exception as qerr:
            # bitsandbytes not available or incompatible — proceed without quantization
            logger.warning(f"bitsandbytes 4-bit quantization unavailable, continuing without quantization: {qerr}")

        # Load model weights as float32 and allow Trainer/Accelerate to
        # perform autocast to float16 during forward passes when fp16 is enabled.
        # Prefer to dispatch the model across GPUs when multiple devices
        # are available to avoid replicating the full model in DataParallel.
        # Attempt to load the model. If the model repo is private or the
        # identifier is incorrect, transformers will raise an OSError indicating
        # missing weight files. In that case, retry using an auth token (if
        # provided via env HF_TOKEN) and provide an actionable error message.
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                model = AutoModel.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id=label2id,
                    dtype=torch.float32,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    **quant_kwargs,
                )
            else:
                model = AutoModel.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id=label2id,
                    dtype=torch.float32,
                    device_map="auto",
                    **quant_kwargs,
                )
        except OSError as oe:
            # Try to retry with an auth token if available. Prefer env var, then config.hf_token.
            import os as _os
            _token = _os.environ.get('HF_TOKEN') or _os.environ.get('HUGGINGFACEHUB_API_TOKEN')
            if not _token:
                try:
                    from config import config as _cfg
                    _token = _cfg.get('hf_token') or _cfg.get('HF_TOKEN')
                except Exception:
                    _token = None
            if _token:
                logger.info("Initial model download failed; retrying with HF token from environment.")
                try:
                    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                        model = AutoModel.from_pretrained(
                            model_name,
                            num_labels=num_labels,
                            id2label=id2label,
                            label2id=label2id,
                            dtype=torch.float32,
                            device_map="auto",
                            low_cpu_mem_usage=True,
                            use_auth_token=_token,
                            **quant_kwargs,
                        )
                    else:
                        model = AutoModel.from_pretrained(
                            model_name,
                            num_labels=num_labels,
                            id2label=id2label,
                            label2id=label2id,
                            dtype=torch.float32,
                            device_map="auto",
                            use_auth_token=_token,
                            **quant_kwargs,
                        )
                except Exception:
                    logger.error("Retry with HF token also failed.")
                    raise
            else:
                # Re-raise with more context
                raise OSError(
                    f"Failed to load model '{model_name}'. {oe}\n"
                    "Possible causes: incorrect model id, the repo has no PyTorch weights, or the repo is private.\n"
                    "If the model is private, set environment variable HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) to a valid token and retry."
                ) from oe
        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=32,
                target_modules=['q_proj','v_proj'], # for mistral
                lora_dropout=0.05,
                bias='none',
                task_type='SEQ_CLS'
            )
            model = get_peft_model(model, lora_config)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Applied LoRA with rank={lora_r}, trainable params: {total_params}")
            if wandb.run is None:
                wandb.log({"trainable_params": total_params, "lora_rank": lora_r})
        if not (hasattr(model, "hf_device_map") and model.hf_device_map is not None):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            logger.info(f"Model moved to device: {device}")
            if wandb.run is not None:
                wandb.log({"device": str(device)})
        else:
            logger.info(f"Model device map: {model.hf_device_map}")
            if wandb.run is not None:
                wandb.log({"device_map": str(model.hf_device_map)})
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise
        
class CustomTrainerWithWeightedLoss(Trainer):
    """
    Custom Trainer to apply class weights to the loss function.
    
    Args: 
        class_weights (torch.Tensor): Tensor of class weights for imbalanced classification.
    """
    def __init__(self, *args, class_weights=None, use_focal_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_focal_loss = use_focal_loss
        if class_weights is not None:
            weights = torch.tensor(list(class_weights.values()), dtype=torch.float32)
            weights = weights / weights.sum() * len(weights)
            self.class_weights = weights.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            if wandb.run is None:
                wandb.log({"class_weights": {i: w.item() for i, w in enumerate(self.class_weights)}})
        else:
            self.class_weights = None
        logger.info(f'Class weights initialized: {self.class_weights}, Focal Loss: {self.use_focal_loss}')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the weighted cross-entropy loss
        
        Args:
            model: The model being trained
            inputs (dict): Input batch including 'labels'
            return_output (bool): Whether to return model outputs 
            
        Returns:
            loss or (loss, outputs)
        """
        try:
            labels = inputs.pop('labels')
            outputs = model(**inputs)
            logits = outputs.logits
            weight = self.class_weights.to(device=logits.device, dtype=logits.dtype) if self.class_weights is not None else None
            labels = labels.to(dtype=torch.long)
            if self.use_focal_loss:
                loss_fnct = FocalLoss(alpha=weight)
                loss = loss_fnct(logits, labels)
            else:
                loss_fnct = torch.nn.CrossEntropyLoss(weight=weight)
                loss = loss_fnct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            if wandb.run is not None:
                wandb.log({"batch_loss": loss.item()})
            return (loss, outputs) if return_outputs else loss
        except Exception as e:
            logger.error(f'Error computing loss: {e}')
            raise