"""
DNABERT Trainer - Complete training and evaluation implementation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from tqdm import tqdm
import time
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, 
                 mode: str = 'min', verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/F1
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.best_score = np.inf
        else:
            self.best_score = -np.inf
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop early.
        
        Args:
            score: Current metric score
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improvement = score < (self.best_score - self.min_delta)
        else:
            improvement = score > (self.best_score + self.min_delta)
        
        if improvement:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                logger.info(f"Early stopping: Best score improved to {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"Early stopping: No improvement for {self.counter} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"Early stopping triggered after {self.patience} epochs")
        
        return self.early_stop


class DNABERTTrainer:
    """
    Complete trainer for DNABERT model with training, validation, and evaluation.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, config: Dict):
        """
        Initialize trainer.
        
        Args:
            model: DNABERT model
            device: Training device
            config: Training configuration
        """
        self.model = model
        self.device = device
        self.config = config
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = None
        
        # Training history
        self.train_history = []
        self.val_history = []
        self.best_model_state = None
        self.best_epoch = 0
        
        # Metrics tracking
        self.best_metrics = {
            'loss': float('inf'),
            'accuracy': 0.0,
            'f1': 0.0,
            'roc_auc': 0.0
        }
        
        logger.info(f"DNABERTTrainer initialized on device: {self.device}")
    
    def _setup_optimizer(self, total_steps: int):
        """Setup optimizer with parameter groups."""
        # Parameters with weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.get('weight_decay', 0.01)
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.get('learning_rate', 2e-5),
            eps=1e-8
        )
        
        # Learning rate scheduler
        warmup_steps = self.config.get('warmup_steps', int(total_steps * 0.1))
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Optimizer setup: lr={self.config.get('learning_rate', 2e-5)}, "
                   f"total_steps={total_steps}, warmup_steps={warmup_steps}")
    
    def _setup_early_stopping(self):
        """Setup early stopping."""
        patience = self.config.get('early_stopping_patience', 5)
        mode = self.config.get('early_stopping_mode', 'max')  # 'max' for F1, 'min' for loss
        min_delta = self.config.get('early_stopping_min_delta', 0.0)
        
        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            mode=mode,
            verbose=True
        )
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('max_grad_norm', 1.0)
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}'
            })
        
        # Calculate metrics
        epoch_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        metrics = {
            'epoch': epoch,
            'train_loss': epoch_loss,
            'train_accuracy': accuracy,
            'train_precision': precision,
            'train_recall': recall,
            'train_f1': f1,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        logger.info(f"Epoch {epoch} - Train: Loss={epoch_loss:.4f}, "
                   f"Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        return metrics
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        val_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(all_labels, [p[1] for p in all_probabilities])
        except:
            roc_auc = 0.0
        
        metrics = {
            'epoch': epoch,
            'val_loss': val_loss,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'val_roc_auc': roc_auc
        }
        
        logger.info(f"Epoch {epoch} - Val: Loss={val_loss:.4f}, "
                   f"Accuracy={accuracy:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, save_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_dir: Directory to save model and results
            
        Returns:
            Training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup components
        total_steps = len(train_loader) * epochs
        self._setup_optimizer(total_steps)
        self._setup_early_stopping()
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Save directory: {save_dir}")
        logger.info(f"Total training steps: {total_steps}")
        
        # Training loop
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            self.val_history.append(val_metrics)
            
            # Check early stopping
            if self.early_stopping is not None:
                if self.early_stopping.mode == 'min':
                    score = val_metrics['val_loss']
                else:
                    score = val_metrics['val_f1']
                
                if self.early_stopping(score):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Save best model
            current_f1 = val_metrics['val_f1']
            if current_f1 > self.best_metrics['f1']:
                self.best_metrics = {
                    'loss': val_metrics['val_loss'],
                    'accuracy': val_metrics['val_accuracy'],
                    'f1': current_f1,
                    'roc_auc': val_metrics['val_roc_auc']
                }
                self.best_epoch = epoch
                self.best_model_state = self.model.state_dict().copy()
                
                # Save best model
                best_model_path = save_dir / "best_model"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'metrics': val_metrics,
                    'config': self.config
                }, best_model_path)
                
                logger.info(f"Best model saved at epoch {epoch} with F1: {current_f1:.4f}")
            
            # Save checkpoint periodically
            if epoch % self.config.get('save_every_n_epochs', 5) == 0:
                checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_history': self.train_history,
                    'val_history': self.val_history,
                    'config': self.config
                }, checkpoint_path)
                
                logger.info(f"Checkpoint saved at epoch {epoch}")
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        # Save final model
        final_model_path = save_dir / "final_model.pt"
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config,
            'best_metrics': self.best_metrics,
            'best_epoch': self.best_epoch
        }, final_model_path)
        
        # Save training history
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'config': self.config
        }
        
        history_path = save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        # Create training report
        self._create_training_report(save_dir)
        
        logger.info(f"Training completed. Best F1: {self.best_metrics['f1']:.4f} "
                   f"at epoch {self.best_epoch}")
        
        return history
    
    def evaluate(self, test_loader: DataLoader, return_predictions: bool = False) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            return_predictions: Whether to return predictions
            
        Returns:
            Evaluation metrics and optionally predictions
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probabilities = []
        all_sequences = []  # Will store if available
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluation")
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        test_loss = total_loss / len(test_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(all_labels, [p[1] for p in all_probabilities])
        except:
            roc_auc = 0.0
        
        # Classification report
        from sklearn.metrics import classification_report
        report = classification_report(all_labels, all_preds, 
                                       target_names=['Non-Promoter', 'Promoter'],
                                       output_dict=True)
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        logger.info(f"Test Results:")
        logger.info(f"  Loss: {test_loss:.4f}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        
        result = {'metrics': metrics}
        
        if return_predictions:
            result['predictions'] = all_preds
            result['probabilities'] = all_probabilities
            result['labels'] = all_labels
        
        return result
    
    def predict(self, sequences: List[str], tokenizer, batch_size: int = 8) -> Dict[str, Any]:
        """
        Predict on new sequences.
        
        Args:
            sequences: List of DNA sequences
            tokenizer: Tokenizer for sequences
            batch_size: Batch size for prediction
            
        Returns:
            Predictions and probabilities
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            # Tokenize batch
            encoding = tokenizer(
                batch_sequences,
                truncation=True,
                padding='max_length',
                max_length=self.config.get('max_length', 512),
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return {
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'sequences': sequences
        }
    
    def _create_training_report(self, save_dir: Path):
        """Create training report with plots."""
        # Convert history to DataFrame
        train_df = pd.DataFrame(self.train_history)
        val_df = pd.DataFrame(self.val_history)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(train_df['epoch'], train_df['train_loss'], label='Train')
        axes[0, 0].plot(val_df['epoch'], val_df['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(train_df['epoch'], train_df['train_accuracy'], label='Train')
        axes[0, 1].plot(val_df['epoch'], val_df['val_accuracy'], label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 plot
        axes[0, 2].plot(train_df['epoch'], train_df['train_f1'], label='Train')
        axes[0, 2].plot(val_df['epoch'], val_df['val_f1'], label='Val')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1-Score')
        axes[0, 2].set_title('Training and Validation F1-Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].plot(train_df['epoch'], train_df['learning_rate'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision-Recall plot
        axes[1, 1].plot(train_df['epoch'], train_df['train_precision'], label='Train Precision')
        axes[1, 1].plot(train_df['epoch'], train_df['train_recall'], label='Train Recall')
        axes[1, 1].plot(val_df['epoch'], val_df['val_precision'], label='Val Precision')
        axes[1, 1].plot(val_df['epoch'], val_df['val_recall'], label='Val Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision and Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # ROC-AUC plot (just value, not actual ROC curve)
        axes[1, 2].plot(val_df['epoch'], val_df['val_roc_auc'])
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('ROC-AUC')
        axes[1, 2].set_title('Validation ROC-AUC')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = save_dir / "training_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {plot_path}")
    
    def save_model(self, path: Union[str, Path], tokenizer=None):
        """
        Save model and tokenizer.
        
        Args:
            path: Path to save to
            tokenizer: Tokenizer to save (optional)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path / "model"
        self.model.save_pretrained(model_path)
        
        # Save tokenizer if provided
        if tokenizer is not None:
            tokenizer_path = path / "tokenizer"
            tokenizer.save_pretrained(tokenizer_path)
        
        # Save training history
        history_path = path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                'train': self.train_history,
                'val': self.val_history,
                'best_epoch': self.best_epoch,
                'best_metrics': self.best_metrics
            }, f, indent=2, default=str)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: Union[str, Path], device: torch.device, 
                   model_class, config: Dict) -> 'DNABERTTrainer':
        """
        Load trainer from saved checkpoint.
        
        Args:
            path: Path to saved checkpoint
            device: Device to load model to
            model_class: Model class to instantiate
            config: Training configuration
            
        Returns:
            Loaded trainer instance
        """
        path = Path(path)
        
        # Load checkpoint
        if path.suffix == '.pt':
            checkpoint = torch.load(path, map_location=device)
            
            # Create model
            model = model_class.from_pretrained(
                checkpoint.get('model_name', 'armheb/DNA_bert_6'),
                num_labels=config.get('num_labels', 2)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            # Create trainer
            trainer = cls(model, device, config)
            trainer.train_history = checkpoint.get('train_history', [])
            trainer.val_history = checkpoint.get('val_history', [])
            trainer.best_epoch = checkpoint.get('best_epoch', 0)
            trainer.best_metrics = checkpoint.get('best_metrics', {})
            
            logger.info(f"Trainer loaded from {path}")
            
            return trainer
        
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


def create_dnabert_trainer(model, device, config_dict: Optional[Dict] = None) -> DNABERTTrainer:
    """
    Factory function to create DNABERT trainer.
    
    Args:
        model: DNABERT model
        device: Training device
        config_dict: Training configuration
        
    Returns:
        DNABERTTrainer instance
    """
    default_config = {
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'warmup_steps': 100,
        'max_grad_norm': 1.0,
        'early_stopping_patience': 5,
        'early_stopping_mode': 'max',
        'early_stopping_min_delta': 0.0,
        'save_every_n_epochs': 5
    }
    
    if config_dict:
        default_config.update(config_dict)
    
    return DNABERTTrainer(model, device, default_config)
