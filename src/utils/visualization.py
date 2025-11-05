"""
Visualization utilities for attention weights and training metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional


def visualize_attention(
    attention_weights: torch.Tensor,
    rna_sequence: str,
    protein_sequence: str,
    output_path: str,
    layer_idx: int = 0,
    head_idx: int = 0
):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights tensor (num_heads, rna_len, protein_len)
        rna_sequence: RNA sequence string
        protein_sequence: Protein sequence string
        output_path: Path to save the visualization
        layer_idx: Layer index to visualize
        head_idx: Attention head index to visualize
    """
    # Get attention for specific head
    if attention_weights.dim() == 4:  # (batch, num_heads, rna_len, protein_len)
        attn = attention_weights[0, head_idx, :, :].cpu().numpy()
    elif attention_weights.dim() == 3:  # (num_heads, rna_len, protein_len)
        attn = attention_weights[head_idx, :, :].cpu().numpy()
    else:
        attn = attention_weights.cpu().numpy()
    
    # Truncate sequences for display if too long
    max_display_len = 100
    rna_display = rna_sequence[:max_display_len] if len(rna_sequence) > max_display_len else rna_sequence
    protein_display = protein_sequence[:max_display_len] if len(protein_sequence) > max_display_len else protein_sequence
    
    # Truncate attention matrix accordingly
    attn_display = attn[:len(rna_display), :len(protein_display)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot heatmap
    sns.heatmap(
        attn_display,
        xticklabels=list(protein_display),
        yticklabels=list(rna_display),
        cmap='viridis',
        ax=ax,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    ax.set_xlabel('Protein Sequence', fontsize=12)
    ax.set_ylabel('RNA Sequence', fontsize=12)
    ax.set_title(f'Cross-Attention Weights (Layer {layer_idx}, Head {head_idx})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved attention visualization to {output_path}")


def plot_training_history(
    history: Dict[str, List],
    output_path: str
):
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training history
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot losses
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax.plot(history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot metrics
    if 'val_metrics' in history and len(history['val_metrics']) > 0:
        # Extract metrics over epochs
        metrics_dict = {}
        for epoch_metrics in history['val_metrics']:
            for key, value in epoch_metrics.items():
                if key not in metrics_dict:
                    metrics_dict[key] = []
                metrics_dict[key].append(value)
        
        # Plot accuracy
        ax = axes[0, 1]
        if 'accuracy' in metrics_dict:
            ax.plot(metrics_dict['accuracy'], label='Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy')
        ax.grid(True, alpha=0.3)
        
        # Plot precision and recall
        ax = axes[1, 0]
        if 'precision' in metrics_dict:
            ax.plot(metrics_dict['precision'], label='Precision')
        if 'recall' in metrics_dict:
            ax.plot(metrics_dict['recall'], label='Recall')
        if 'f1' in metrics_dict:
            ax.plot(metrics_dict['f1'], label='F1-Score')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Validation Precision, Recall, and F1')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot AUC
        ax = axes[1, 1]
        if 'auc' in metrics_dict:
            ax.plot(metrics_dict['auc'], label='AUC-ROC')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.set_title('Validation AUC-ROC')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training history plot to {output_path}")


def plot_roc_curve(
    labels: np.ndarray,
    predictions: np.ndarray,
    output_path: str
):
    """
    Plot ROC curve.
    
    Args:
        labels: True labels
        predictions: Predicted probabilities
        output_path: Path to save the plot
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ROC curve to {output_path}")


def visualize_binding_sites(
    attention_weights: torch.Tensor,
    rna_sequence: str,
    protein_sequence: str,
    output_path: str,
    top_k: int = 10
):
    """
    Visualize top binding sites based on attention weights.
    
    Args:
        attention_weights: Attention weights (num_heads, rna_len, protein_len)
        rna_sequence: RNA sequence string
        protein_sequence: Protein sequence string
        output_path: Path to save the visualization
        top_k: Number of top binding sites to show
    """
    # Average over attention heads
    if attention_weights.dim() == 4:  # (batch, num_heads, rna_len, protein_len)
        attn = attention_weights[0].mean(dim=0).cpu().numpy()
    elif attention_weights.dim() == 3:  # (num_heads, rna_len, protein_len)
        attn = attention_weights.mean(dim=0).cpu().numpy()
    else:
        attn = attention_weights.cpu().numpy()
    
    # Find top-k RNA positions with highest attention
    rna_attention = attn.sum(axis=1)  # Sum over protein positions
    top_rna_indices = np.argsort(rna_attention)[-top_k:][::-1]
    
    # Find top-k protein positions with highest attention
    protein_attention = attn.sum(axis=0)  # Sum over RNA positions
    top_protein_indices = np.argsort(protein_attention)[-top_k:][::-1]
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # RNA binding sites
    ax = axes[0]
    ax.bar(range(top_k), rna_attention[top_rna_indices])
    ax.set_xlabel('RNA Position')
    ax.set_ylabel('Total Attention Weight')
    ax.set_title(f'Top {top_k} RNA Binding Sites')
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"{idx}\n{rna_sequence[idx]}" for idx in top_rna_indices])
    ax.grid(True, alpha=0.3)
    
    # Protein binding sites
    ax = axes[1]
    ax.bar(range(top_k), protein_attention[top_protein_indices])
    ax.set_xlabel('Protein Position')
    ax.set_ylabel('Total Attention Weight')
    ax.set_title(f'Top {top_k} Protein Binding Sites')
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"{idx}\n{protein_sequence[idx] if idx < len(protein_sequence) else 'PAD'}" 
                         for idx in top_protein_indices])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved binding sites visualization to {output_path}")
