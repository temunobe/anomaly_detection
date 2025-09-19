# roberta_model.py
import torch
import logging
import wandb
from transformers import RobertaForSequenceClassification, Trainer
from peft import LoraConfig, get_peft_model

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocalLoss(torch.nn.Module):
    """Focal loss for addressing class imbalance."""
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal

def init_roberta_model(
    model_name, 
    num_labels=2, 
    id2label=None, 
    label2id=None, 
    dropout=0.1, 
    use_lora=True, 
    lora_r=16
):
    """
    Initialize RoBERTa model for sequence classification
    
    Args:
        model_name (str): Name or path of the pretrained RoBERTa model (default: 'roberta-base')
        num_labels (int): Number of output labels (e.g., 2 for binary anomaly detection)
        id2label (dict, optional): Mapping from label IDs to label names 
        label2id (dict, optional): Mapping from label names to label IDs
        dropout (float, optional): Custom dropout rate for classifier head (default: 0.1)
        use_lora (bool): Whether to apply LoRA for parameter-efficient fine-tuning (default: True)
        lora_r (int): LoRA rank parameter (default: 16)
    Returns: 
        RobertaForSequenceClassification: Initialized model, optionally with LoRA
    """
    
    logging.info(f"Initializing RoBERTa model: {model_name} with {num_labels} labels, LoRA={use_lora}")
    try:
        model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model.config.hidden_dropout_prob = dropout
        model.gradient_checkpointing_enable()
        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=32,
                target_modules=['query', 'value'],
                lora_dropout=0.05,
                bias='none',
                task_type='SEQ_CLS'
            )
            model = get_peft_model(model, lora_config)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'Applied LoRA with rank={lora_r}, trainable params: {total_params}')
            if wandb.run is not None:
                wandb.log({"trainable_params": total_params, "lora_rank": lora_r})
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        logger.info(f'Model moved to device: {device}')
        if wandb.run is not None:
            wandb.log({"device": str(device)})
        return model
    except Exception as e:
        logger.error(f'Failed to initialize model: {e}')
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

