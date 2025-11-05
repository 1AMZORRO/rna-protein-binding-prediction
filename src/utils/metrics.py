"""
Metrics calculation utilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from typing import Dict


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: Predicted probabilities
        labels: True labels
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    # Convert probabilities to binary predictions
    binary_preds = (predictions >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(labels, binary_preds),
        'precision': precision_score(labels, binary_preds, zero_division=0),
        'recall': recall_score(labels, binary_preds, zero_division=0),
        'f1': f1_score(labels, binary_preds, zero_division=0)
    }
    
    # Calculate AUC-ROC if possible
    try:
        metrics['auc'] = roc_auc_score(labels, predictions)
    except ValueError:
        metrics['auc'] = 0.0
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, binary_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
    
    return metrics


def evaluate_model(model, data_loader, device='cuda'):
    """
    Evaluate model on a dataset.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset
        device: Device to use
        
    Returns:
        Dictionary of predictions, labels, and metrics
    """
    import torch
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            rna_batch, protein_batch, protein_mask, labels = batch
            
            # Move to device
            rna_batch = rna_batch.to(device)
            protein_batch = protein_batch.to(device)
            protein_mask = protein_mask.to(device)
            
            # Forward pass
            outputs, _ = model(rna_batch, protein_batch, protein_mask)
            
            # Store predictions and labels
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels)
    
    return {
        'predictions': predictions,
        'labels': labels,
        'metrics': metrics
    }


def print_metrics(metrics: Dict):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*50)
    print("Evaluation Metrics")
    print("="*50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key.capitalize()}: {value:.4f}")
        else:
            print(f"{key.capitalize()}: {value}")
    
    print("="*50 + "\n")
