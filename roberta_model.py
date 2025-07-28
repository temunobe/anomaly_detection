# roberta_model.py
import torch
import logging
from transformers import RobertaForSequenceClassification, Trainer

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_roberta_model(model_name, num_labels, id2label=None, label2id=None, dropout=None):
    """
    Initialize RoBERTa model for sequence classification
    
    Args:
        model_name (str): Name or path of the pretrained RoBERTa model.
        num_labels (int): Number of output labels
        id2label (dict, optional): Mapping from label IDs to label names
        label2id (dict, optional): Mapping from label names to label IDs
        dropout (float, optional): Custom dropout rate for classifier head
        
    Returns: 
        RobertaForSequenceClassification: Initialized model
    """
    
    logging.info(f"Initializing RoBERTa model: {model_name} with {num_labels} labels")
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=dropout if dropout is not None else 0.1
    )

    return model

class CustomTrainerWithWeightedLoss(Trainer):
    """
    Custom Trainer to apply class weights to the loss function.
    
    Args: 
        class_weights (torch.Tensor): Tensor of class weights for imbalanced classification.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the weighted cross-entropy loss
        
        Args:
            model: The model being trained
            inputs (dict): Input batch including 'labels'
            return_output (bool): Whether to return model outputs 
            
        Returns:
            loss or (loss, outputs)
        """
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits#get('logits')

        weights_tensor = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fnct = torch.nn.CrossEntropyLoss(weight=weights_tensor)
        loss = loss_fnct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

