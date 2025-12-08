import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class DNABERTTrainer:
    def __init__(self, model: nn.Module, device: torch.device, config: Dict):
        self.model = model
        self.device = device
        self.config = config
        self.optimizer = None
        self.scheduler = None
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            self.optimizer.zero_grad()
            
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        return {
            'epoch': epoch,
            'train_loss': epoch_loss,
            'train_accuracy': accuracy,
            'train_f1': f1
        }
    
    def evaluate(self, val_loader: DataLoader) -> Dict:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
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
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        return {
            'val_loss': val_loss,
            'val_accuracy': accuracy,
            'val_f1': f1
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, save_dir: str):
        """Full training loop"""
        # Setup optimizer
        total_steps = len(train_loader) * epochs
        self.optimizer = AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        best_f1 = 0
        history = []
        
        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            history.append(epoch_metrics)
            
            print(f"Epoch {epoch}: "
                  f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Val F1: {val_metrics['val_f1']:.4f}")
            
            # Save best model
            if val_metrics['val_f1'] > best_f1:
                best_f1 = val_metrics['val_f1']
                torch.save(self.model.state_dict(), f"{save_dir}/best_model.pt")
                print(f"Best model saved with F1: {best_f1:.4f}")
        
        return history
