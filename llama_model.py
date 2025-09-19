# llama_model.py

import torch
import logging
import wandb

from transformers import LlamaForSequenceClassification, LlamaTokenizerFast
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
        
def init_llama_model(
    model_name,
    num_labels=2,
    id2label=None,
    label2id=None,
    use_lora=True,
    lora_r=16
):
    """
    Initializing Llama model for sequence classifitication
    """
    logging.info(f"Initializing Llama model: {model_name} with {num_labels} labels, LoRA={use_lora}")
    try:
        model = LlamaForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels, 
            id2label=id2label,
            label2id=label2id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=32,
                target_modules=['q_proj','v_proj'], # for llama
                lora_dropout=0.05,
                bias='none',
                task_type='SEQ_CLS'
            )
            model = get_peft_model(model, lora_config)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Applied LoRA with rank={lora_r}, trainable params: {total_params}")
            if wandb.run is None:
                wandb.log({"trainable_params": total_params, "lora_rank": lora_r})
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        logger.info(f"Model moved to device: {device}")
        if wandb.run is not None:
            wandb.log({"device": str(device)})
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise