"""
Training loop for RNA-Protein binding prediction model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json


class Trainer:
    """Trainer class for RNA-Protein binding prediction."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = "cuda",
        checkpoint_dir: str = "models/checkpoints",
        early_stopping_patience: int = 15,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        grad_clip: float = 1.0
    ):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            device: Device to use
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Early stopping patience
            scheduler_patience: Learning rate scheduler patience
            scheduler_factor: Learning rate reduction factor
            grad_clip: Gradient clipping value
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True
        )
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.early_stopping_patience = early_stopping_patience
        self.grad_clip = grad_clip
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            rna_batch, protein_batch, protein_mask, labels = batch
            
            # Move to device
            rna_batch = rna_batch.to(self.device)
            protein_batch = protein_batch.to(self.device)
            protein_mask = protein_mask.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, _ = self.model(rna_batch, protein_batch, protein_mask)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> Tuple[float, Dict]:
        """Validate the model."""
        if self.val_loader is None:
            return 0.0, {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                rna_batch, protein_batch, protein_mask, labels = batch
                
                # Move to device
                rna_batch = rna_batch.to(self.device)
                protein_batch = protein_batch.to(self.device)
                protein_mask = protein_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(rna_batch, protein_batch, protein_mask)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and labels
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_labels)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, predictions, labels, threshold=0.5):
        """Calculate evaluation metrics."""
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Binary predictions
        binary_preds = (predictions >= threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((binary_preds == 1) & (labels == 1))
        tn = np.sum((binary_preds == 0) & (labels == 0))
        fp = np.sum((binary_preds == 1) & (labels == 0))
        fn = np.sum((binary_preds == 0) & (labels == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate AUC-ROC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels, predictions)
        except:
            auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return metrics
    
    def train(self, num_epochs: int, save_best_only: bool = True):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_best_only: Whether to save only the best model
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            if self.val_loader is not None:
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val Metrics: {val_metrics}")
            
            # Learning rate scheduling
            if self.val_loader is not None:
                self.scheduler.step(val_loss)
            
            # Save checkpoint
            if self.val_loader is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                if save_best_only:
                    self.save_checkpoint(epoch, val_loss, val_metrics, is_best=True)
                    print(f"Saved best model with val_loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            if not save_best_only:
                self.save_checkpoint(epoch, val_loss, val_metrics, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print("\nTraining completed!")
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(self, epoch: int, val_loss: float, val_metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'history': self.history
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
    
    def save_history(self):
        """Save training history to JSON file."""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {history_path}")
