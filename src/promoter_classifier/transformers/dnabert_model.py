"""
DNABERT Model Wrapper - Complete implementation
"""

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

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DNABERTTrainingConfig:
    """Configuration for DNABERT model training."""
    model_name: str = "armheb/DNA_bert_6"
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    num_labels: int = 2
    output_attentions: bool = True
    output_hidden_states: bool = True
    use_class_weights: bool = False
    early_stopping_patience: int = 5
    save_every_n_epochs: int = 5
    random_state: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_labels < 2:
            raise ValueError("num_labels must be at least 2")
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'DNABERTTrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: Union[str, Path]):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Config saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DNABERTTrainingConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class DNABERTModel:
    """
    Complete wrapper for DNABERT model with training and inference capabilities.
    """
    
    def __init__(self, config: Optional[DNABERTTrainingConfig] = None, device: Optional[str] = None):
        """
        Initialize DNABERT model wrapper.
        
        Args:
            config: Training configuration
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.config = config or DNABERTTrainingConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.class_weights = None
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
        logger.info(f"DNABERTModel initialized with device: {self.device}")
        logger.info(f"Config: {self.config}")
    
    def _setup_tokenizer(self):
        """Setup tokenizer for DNA sequences."""
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.config.model_name)
            
            # Add special tokens if needed
            special_tokens_dict = {
                'additional_special_tokens': ['[PROM]', '[NON_PROM]']
            }
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
            
            logger.info(f"Tokenizer loaded. Vocab size: {self.tokenizer.vocab_size}")
            logger.info(f"Added {num_added_toks} special tokens")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise
    
    def _setup_model(self):
        """Setup DNABERT model architecture."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        try:
            self.model = BertForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states
            )
            
            # Resize token embeddings if we added special tokens
            if self.tokenizer is not None:
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Move model to device
            self.model.to(self.device)
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Model device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def load(self):
        """
        Load tokenizer and model.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if self.tokenizer is None:
            self._setup_tokenizer()
        if self.model is None:
            self._setup_model()
        
        return self.model, self.tokenizer
    
    def tokenize_sequence(self, sequence: str, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize a single DNA sequence.
        
        Args:
            sequence: DNA sequence string
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        if self.tokenizer is None:
            self._setup_tokenizer()
        
        max_length = max_length or self.config.max_length
        sequence = str(sequence).upper()  # DNABERT expects uppercase
        
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
    
    def tokenize_batch(self, sequences: List[str], max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of DNA sequences.
        
        Args:
            sequences: List of DNA sequences
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with batched 'input_ids' and 'attention_mask'
        """
        if self.tokenizer is None:
            self._setup_tokenizer()
        
        max_length = max_length or self.config.max_length
        sequences = [str(seq).upper() for seq in sequences]
        
        encoding = self.tokenizer(
            sequences,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        return encoding
    
    def predict_single(self, sequence: str, return_attention: bool = False) -> Dict:
        """
        Make prediction for a single sequence.
        
        Args:
            sequence: DNA sequence
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        self.model.eval()
        
        # Tokenize
        encoding = self.tokenize_sequence(sequence)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=return_attention
            )
        
        # Process outputs
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1)
        
        result = {
            'sequence': sequence,
            'prediction': prediction.item(),
            'probabilities': probabilities.cpu().numpy()[0].tolist(),
            'logits': logits.cpu().numpy()[0].tolist()
        }
        
        if return_attention and outputs.attentions is not None:
            result['attentions'] = [att.cpu().numpy() for att in outputs.attentions]
        
        return result
    
    def predict_batch(self, sequences: List[str], batch_size: int = 8, 
                     return_attention: bool = False) -> Dict:
        """
        Make predictions for a batch of sequences.
        
        Args:
            sequences: List of DNA sequences
            batch_size: Batch size for inference
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with batch results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_sequences = []
        all_attentions = [] if return_attention else None
        
        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            # Tokenize batch
            encoding = self.tokenize_batch(batch_sequences)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=return_attention
                )
            
            # Process outputs
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_sequences.extend(batch_sequences)
            
            if return_attention and outputs.attentions is not None:
                batch_attentions = [att.cpu().numpy() for att in outputs.attentions]
                all_attentions.append(batch_attentions)
        
        result = {
            'sequences': all_sequences,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'num_samples': len(sequences)
        }
        
        if return_attention and all_attentions:
            result['attentions'] = all_attentions
        
        return result
    
    def save(self, save_dir: Union[str, Path], save_tokenizer: bool = True):
        """
        Save model, tokenizer, and config to directory.
        
        Args:
            save_dir: Directory to save to
            save_tokenizer: Whether to save tokenizer
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.model is not None:
            model_path = save_dir / "model"
            self.model.save_pretrained(model_path)
            logger.info(f"Model saved to {model_path}")
        
        # Save tokenizer
        if save_tokenizer and self.tokenizer is not None:
            tokenizer_path = save_dir / "tokenizer"
            self.tokenizer.save_pretrained(tokenizer_path)
            logger.info(f"Tokenizer saved to {tokenizer_path}")
        
        # Save config
        config_path = save_dir / "config.json"
        self.config.save(config_path)
        
        # Save training history if available
        if self.training_history:
            history_path = save_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            logger.info(f"Training history saved to {history_path}")
        
        logger.info(f"All components saved to {save_dir}")
    
    def load_from_dir(self, load_dir: Union[str, Path]):
        """
        Load model, tokenizer, and config from directory.
        
        Args:
            load_dir: Directory to load from
        """
        load_dir = Path(load_dir)
        
        # Load config
        config_path = load_dir / "config.json"
        if config_path.exists():
            self.config = DNABERTTrainingConfig.load(config_path)
            logger.info(f"Config loaded from {config_path}")
        
        # Load tokenizer
        tokenizer_path = load_dir / "tokenizer"
        if tokenizer_path.exists():
            self.tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))
            logger.info(f"Tokenizer loaded from {tokenizer_path}")
        
        # Load model
        model_path = load_dir / "model"
        if model_path.exists():
            self.model = BertForSequenceClassification.from_pretrained(
                str(model_path),
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states
            )
            self.model.to(self.device)
            self.is_trained = True
            logger.info(f"Model loaded from {model_path}")
        
        # Load training history if available
        history_path = load_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
            logger.info(f"Training history loaded from {history_path}")
    
    def get_attention_weights(self, sequence: str, layer_idx: int = -1) -> np.ndarray:
        """
        Get attention weights for a sequence.
        
        Args:
            sequence: DNA sequence
            layer_idx: Layer index (-1 for last layer)
            
        Returns:
            Attention weights matrix
        """
        result = self.predict_single(sequence, return_attention=True)
        
        if 'attentions' not in result:
            raise ValueError("Model was not configured to output attention weights")
        
        attentions = result['attentions']
        
        if layer_idx < 0:
            layer_idx = len(attentions) + layer_idx
        
        if layer_idx >= len(attentions):
            raise ValueError(f"Layer index {layer_idx} out of range. Model has {len(attentions)} layers")
        
        # Get attention for the first head of the specified layer
        attention_matrix = attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]
        
        # Average across heads
        avg_attention = attention_matrix.mean(axis=0)
        
        return avg_attention
    
    def compute_class_weights(self, labels: List[int]) -> torch.Tensor:
        """
        Compute class weights for imbalanced datasets.
        
        Args:
            labels: List of class labels
            
        Returns:
            Tensor of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=labels
        )
        
        self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        logger.info(f"Class weights computed: {class_weights}")
        
        return self.class_weights


def create_default_dnabert_config() -> DNABERTTrainingConfig:
    """
    Create default DNABERT training configuration.
    
    Returns:
        Default configuration
    """
    return DNABERTTrainingConfig()


def create_dnabert_for_promoter_classification(device: Optional[str] = None) -> DNABERTModel:
    """
    Factory function to create DNABERT model for promoter classification.
    
    Args:
        device: Device to run model on
        
    Returns:
        Configured DNABERT model
    """
    config = DNABERTTrainingConfig(
        model_name="armheb/DNA_bert_6",
        max_length=512,  # DNABERT works best with longer sequences
        batch_size=8,    # Smaller batch size due to longer sequences
        num_epochs=10,
        learning_rate=2e-5,
        num_labels=2,
        output_attentions=True,
        output_hidden_states=True
    )
    
    model = DNABERTModel(config=config, device=device)
    return model
