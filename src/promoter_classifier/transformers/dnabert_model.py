import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer

class DNABERTWrapper:
    def __init__(self, model_name="armheb/DNA_bert_6", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
    
    def load(self, num_labels=2):
        """Загрузка модели и токенизатора"""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            output_attentions=True
        )
        self.model.to(self.device)
        return self.model, self.tokenizer
