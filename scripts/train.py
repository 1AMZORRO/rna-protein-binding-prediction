"""
Training script for RNA-Protein binding prediction model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split
import random
import numpy as np

from src.data import RNAProcessor, ProteinProcessor, RNAProteinDataset
from src.data.dataset import collate_fn
from src.models import RNAProteinBindingModel
from src.training import Trainer
from src.utils import plot_training_history


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(config, rna_processor, protein_processor):
    """
    Create training, validation, and test data loaders.
    
    This is a placeholder function. In practice, you would load your data here.
    """
    # For demonstration, we create dummy data
    # In practice, replace this with actual data loading
    print("Creating dummy dataset for demonstration...")
    
    # Generate dummy RNA sequences (101 bp)
    rna_seqs = []
    protein_seqs = []
    labels = []
    
    nucleotides = ['A', 'C', 'G', 'U']
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Create 100 positive and 100 negative samples
    for i in range(200):
        # Random RNA sequence
        rna_seq = ''.join(random.choice(nucleotides) for _ in range(101))
        rna_seqs.append(rna_seq)
        
        # Random protein sequence
        protein_len = random.randint(50, 200)
        protein_seq = ''.join(random.choice(amino_acids) for _ in range(protein_len))
        protein_seqs.append(protein_seq)
        
        # Random label
        labels.append(random.randint(0, 1))
    
    # Create dataset
    dataset = RNAProteinDataset(
        rna_sequences=rna_seqs,
        protein_sequences=protein_seqs,
        labels=labels,
        rna_processor=rna_processor,
        protein_processor=protein_processor
    )
    
    # Split dataset
    train_size = int(config['data']['train_split'] * len(dataset))
    val_size = int(config['data']['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=config['data']['shuffle'],
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='Train RNA-Protein binding prediction model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--rna-fasta', type=str, default=None,
                        help='Path to RNA FASTA file')
    parser.add_argument('--protein-fasta', type=str, default=None,
                        help='Path to protein FASTA file')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to labels file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.output_dir:
        config['paths']['checkpoint_dir'] = args.output_dir
    
    # Set random seed
    set_seed(config['seed'])
    
    # Set device
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directories
    Path(config['paths']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Initialize processors
    print("Initializing RNA and Protein processors...")
    rna_processor = RNAProcessor(seq_length=config['model']['rna_seq_length'])
    
    # Note: ESM2 model loading can take some time and requires internet connection
    # Use a smaller model for faster initialization during development
    # For offline mode, download the model first and set local_esm_model_path in config
    local_model_path = config['model'].get('local_esm_model_path', None)
    protein_processor = ProteinProcessor(
        model_name=config['model']['esm_model_name'],
        device=device,
        local_model_path=local_model_path
    )
    
    # Create data loaders
    print("Creating data loaders...")
    if args.rna_fasta and args.protein_fasta:
        # Load from FASTA files
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
        
        # Split dataset
        train_size = int(config['data']['train_split'] * len(dataset))
        val_size = int(config['data']['val_split'] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'],
                                  shuffle=True, num_workers=0, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'],
                                shuffle=False, num_workers=0, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'],
                                 shuffle=False, num_workers=0, collate_fn=collate_fn)
    else:
        # Use dummy data for demonstration
        train_loader, val_loader, test_loader = create_data_loaders(
            config, rna_processor, protein_processor
        )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = RNAProteinBindingModel(
        rna_input_dim=config['model']['rna_input_dim'],
        protein_input_dim=config['model']['protein_embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_attention_heads=config['model']['num_attention_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        rna_seq_length=config['model']['rna_seq_length']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        device=device,
        checkpoint_dir=config['paths']['checkpoint_dir'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        scheduler_patience=config['training']['scheduler_patience'],
        scheduler_factor=config['training']['scheduler_factor'],
        grad_clip=config['training']['grad_clip']
    )
    
    # Train model
    print("\nStarting training...")
    trainer.train(
        num_epochs=config['training']['epochs'],
        save_best_only=config['save_best_only']
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(
        trainer.history,
        output_path=str(Path(config['paths']['output_dir']) / 'training_history.png')
    )
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
