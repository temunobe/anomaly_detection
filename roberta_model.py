# from tensorflow.keras.utils import to_categorical # Might not be needed if RoBERTa takes integer labels
import torch
from transformers import RobertaForSequenceClassification, Trainer

def init_roberta_model(model_name, num_labels, id2label=None, label2id=None):
    """Initialize RoBERTa model for sequence classification"""
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label, # For better model checkpoint readability
        label2id=label2id
    )

    return model

class CustomTrainerWithWeightedLoss(Trainer):
    """Custom Trainer to apply class weights to the loss function."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')

        loss = None
        if labels is not None:
            # Move class_weights to the same device as logits
            weights_tensor = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss_fnct = torch.nn.CrossEntropyLoss(weight=weights_tensor)
            loss = loss_fnct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

