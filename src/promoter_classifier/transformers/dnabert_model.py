import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import os
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DNABERTConfig:
    """Configuration for DNABERT model"""
    model_name: str = "armheb/DNA_bert_6"
    max_length: int = 512
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 10
    num_labels: int = 2
    output_attentions: bool = True
    
    def validate(self):
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

class DNABERTModel:
    def __init__(self, config: Optional[DNABERTConfig] = None, device: Optional[str] = None):
        self.config = config or DNABERTConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
    def load_tokenizer(self):
        """Load DNABERT tokenizer"""
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(self.config.model_name)
        return self.tokenizer
    
    def load_model(self):
        """Load DNABERT model"""
        logger.info(f"Loading model: {self.config.model_name}")
        self.model = BertForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            output_attentions=self.config.output_attentions
        )
        self.model.to(self.device)
        return self.model
    
    def tokenize_sequence(self, sequence: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single DNA sequence"""
        if self.tokenizer is None:
            self.load_tokenizer()
        
        encoding = self.tokenizer(
            sequence.upper(),
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        return encoding
    
    def predict_single(self, sequence: str) -> Dict:
        """Predict for single sequence"""
        if self.model is None:
            self.load_model()
        
        self.model.eval()
        encoding = self.tokenize_sequence(sequence)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'].to(self.device),
                attention_mask=encoding['attention_mask'].to(self.device)
            )
        
        probabilities = torch.softmax(outputs.logits, dim=-1)
        
        return {
            'prediction': torch.argmax(outputs.logits).item(),
            'probabilities': probabilities.cpu().numpy()[0].tolist(),
            'logits': outputs.logits.cpu().numpy()[0].tolist()
        }
    
    def save(self, save_dir: str):
        """Save model and tokenizer"""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.model:
            self.model.save_pretrained(os.path.join(save_dir, "model"))
        if self.tokenizer:
            self.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
        
        # Save config
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    def load_from_dir(self, model_dir: str):
        """Load from saved directory"""
        # Load config
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = DNABERTConfig(**config_dict)
        
        # Load tokenizer and model
        tokenizer_path = os.path.join(model_dir, "tokenizer")
        model_path = os.path.join(model_dir, "model")
        
        if os.path.exists(tokenizer_path):
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        if os.path.exists(model_path):
            self.model = BertForSequenceClassification.from_pretrained(
                model_path,
                output_attentions=self.config.output_attentions
            )
            self.model.to(self.device)
