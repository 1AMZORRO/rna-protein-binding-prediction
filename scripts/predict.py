"""
Prediction script for RNA-Protein binding prediction model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from src.data import RNAProcessor, ProteinProcessor, RNAProteinDataset
from src.data.dataset import collate_fn
from src.models import RNAProteinBindingModel
from src.utils import visualize_attention, calculate_metrics, print_metrics, visualize_binding_sites


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, config: dict, device: str):
    """Load trained model from checkpoint."""
    model = RNAProteinBindingModel(
        rna_input_dim=config['model']['rna_input_dim'],
        protein_input_dim=config['model']['protein_embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_attention_heads=config['model']['num_attention_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        rna_seq_length=config['model']['rna_seq_length']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    if 'val_loss' in checkpoint:
        print(f"Checkpoint validation loss: {checkpoint['val_loss']:.4f}")
    
    return model


def predict(
    model,
    data_loader,
    device: str,
    return_attention: bool = False
):
    """
    Make predictions on dataset.
    
    Args:
        model: Trained model
        data_loader: DataLoader for the dataset
        device: Device to use
        return_attention: Whether to return attention weights
        
    Returns:
        Predictions and optionally attention weights
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_attention_weights = [] if return_attention else None
    
    with torch.no_grad():
        for batch in data_loader:
            rna_batch, protein_batch, protein_mask, labels = batch
            
            # Move to device
            rna_batch = rna_batch.to(device)
            protein_batch = protein_batch.to(device)
            protein_mask = protein_mask.to(device)
            
            # Forward pass
            outputs, attention_weights = model(rna_batch, protein_batch, protein_mask)
            
            # Store results
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            if return_attention:
                # Store attention from last layer
                all_attention_weights.append(attention_weights[-1].cpu())
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    if return_attention:
        return predictions, labels, all_attention_weights
    else:
        return predictions, labels


def main():
    parser = argparse.ArgumentParser(description='Predict RNA-Protein binding')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--rna-fasta', type=str, required=True,
                        help='Path to RNA FASTA file')
    parser.add_argument('--protein-fasta', type=str, required=True,
                        help='Path to protein FASTA file')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to labels file (optional, for evaluation)')
    parser.add_argument('--output', type=str, default='predictions.txt',
                        help='Output file for predictions')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate attention visualizations')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for visualizations')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for prediction')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Set device
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processors
    print("Initializing RNA and Protein processors...")
    rna_processor = RNAProcessor(seq_length=config['model']['rna_seq_length'])
    local_model_path = config['model'].get('local_esm_model_path', None)
    protein_processor = ProteinProcessor(
        model_name=config['model']['esm_model_name'],
        device=device,
        local_model_path=local_model_path
    )
    
    # Load data
    print(f"Loading data from {args.rna_fasta} and {args.protein_fasta}")
    labels_list = None
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_list = [int(line.strip()) for line in f]
    
    dataset = RNAProteinDataset.from_fasta_files(
        rna_fasta=args.rna_fasta,
        protein_fasta=args.protein_fasta,
        labels=labels_list,
        rna_processor=rna_processor,
        protein_processor=protein_processor
    )
    
    batch_size = args.batch_size if args.batch_size else config['data']['batch_size']
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"Loaded {len(dataset)} samples")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model, config, device)
    
    # Make predictions
    print("Making predictions...")
    if args.visualize:
        predictions, labels, attention_weights = predict(
            model, data_loader, device, return_attention=True
        )
    else:
        predictions, labels = predict(model, data_loader, device, return_attention=False)
    
    # Save predictions
    print(f"Saving predictions to {args.output}")
    output_data = {
        'prediction': predictions,
    }
    if labels_list is not None:
        output_data['label'] = labels
        output_data['correct'] = (predictions >= 0.5).astype(int) == labels
    
    df = pd.DataFrame(output_data)
    df.to_csv(args.output, sep='\t', index=False)
    
    # Calculate and print metrics if labels are available
    if labels_list is not None:
        print("\nCalculating metrics...")
        metrics = calculate_metrics(predictions, labels)
        print_metrics(metrics)
        
        # Save metrics
        metrics_file = output_dir / 'metrics.txt'
        with open(metrics_file, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        print(f"Saved metrics to {metrics_file}")
    
    # Generate visualizations
    if args.visualize and attention_weights:
        print("\nGenerating visualizations...")
        
        # Visualize attention for first few samples
        num_visualizations = min(5, len(dataset))
        
        for i in range(num_visualizations):
            rna_seq = dataset.rna_sequences[i]
            protein_seq = dataset.protein_sequences[i]
            
            # Get attention for this sample
            batch_idx = i // batch_size
            sample_idx = i % batch_size
            
            if batch_idx < len(attention_weights):
                attn = attention_weights[batch_idx][sample_idx]
                
                # Visualize attention heatmap
                visualize_attention(
                    attn,
                    rna_seq,
                    protein_seq,
                    output_path=str(output_dir / f'attention_sample_{i}.png'),
                    layer_idx=0,
                    head_idx=0
                )
                
                # Visualize binding sites
                visualize_binding_sites(
                    attn,
                    rna_seq,
                    protein_seq,
                    output_path=str(output_dir / f'binding_sites_sample_{i}.png'),
                    top_k=10
                )
        
        print(f"Saved visualizations to {output_dir}")
    
    print("\nPrediction completed successfully!")


if __name__ == '__main__':
    main()
