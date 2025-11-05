"""
Basic tests for RNA-Protein binding prediction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.data import RNAProcessor, RNAProteinDataset
from src.models import RNAProteinBindingModel


def test_rna_processor():
    """Test RNA sequence processing."""
    print("\n=== Testing RNA Processor ===")
    
    processor = RNAProcessor(seq_length=101)
    
    # Test single sequence encoding
    sequence = "ACGUACGUACGU"
    encoding = processor.encode_sequence(sequence)
    
    assert encoding.shape == (101, 4), f"Expected shape (101, 4), got {encoding.shape}"
    assert encoding.dtype == torch.float32
    print("✓ Single sequence encoding works")
    
    # Test batch encoding
    sequences = ["ACGU" * 25 + "A", "UGCA" * 25 + "U"]
    batch = processor.encode_batch(sequences)
    
    assert batch.shape == (2, 101, 4), f"Expected shape (2, 101, 4), got {batch.shape}"
    print("✓ Batch encoding works")
    
    # Test decoding
    decoded = processor.decode_sequence(encoding)
    assert len(decoded) == 101
    print("✓ Sequence decoding works")
    
    print("✓ RNA Processor tests passed!\n")


def test_model_forward():
    """Test model forward pass."""
    print("\n=== Testing Model Forward Pass ===")
    
    model = RNAProteinBindingModel(
        rna_input_dim=4,
        protein_input_dim=1280,
        hidden_dim=128,
        num_attention_heads=4,
        num_layers=2,
        dropout=0.1,
        rna_seq_length=101
    )
    
    # Create dummy inputs
    batch_size = 2
    rna_emb = torch.randn(batch_size, 101, 4)
    protein_emb = torch.randn(batch_size, 50, 1280)
    protein_mask = torch.ones(batch_size, 50)
    
    # Forward pass
    output, attention_weights = model(rna_emb, protein_emb, protein_mask)
    
    assert output.shape == (batch_size,), f"Expected output shape ({batch_size},), got {output.shape}"
    assert len(attention_weights) == 2, f"Expected 2 attention layers, got {len(attention_weights)}"
    assert torch.all((output >= 0) & (output <= 1)), "Output should be in [0, 1] range"
    
    print(f"✓ Model output shape: {output.shape}")
    print(f"✓ Number of attention layers: {len(attention_weights)}")
    print(f"✓ Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("✓ Model forward pass tests passed!\n")


def test_dataset():
    """Test dataset creation."""
    print("\n=== Testing Dataset ===")
    
    rna_seqs = ["ACGU" * 25 + "A", "UGCA" * 25 + "U", "GCAU" * 25 + "G"]
    protein_seqs = ["ACDEFGHIKL" * 5, "LMNPQRSTVW" * 5, "ACDEFGHIKL" * 4]
    labels = [1, 0, 1]
    
    processor = RNAProcessor(seq_length=101)
    
    dataset = RNAProteinDataset(
        rna_sequences=rna_seqs,
        protein_sequences=protein_seqs,
        labels=labels,
        rna_processor=processor,
        protein_processor=None  # Skip protein processor for speed
    )
    
    assert len(dataset) == 3, f"Expected dataset length 3, got {len(dataset)}"
    
    # Test single item
    rna_emb, protein_emb, label = dataset[0]
    assert rna_emb.shape == (101, 4)
    assert label == 1
    
    print(f"✓ Dataset length: {len(dataset)}")
    print(f"✓ Sample RNA embedding shape: {rna_emb.shape}")
    print("✓ Dataset tests passed!\n")


def test_model_parameters():
    """Test model parameter count."""
    print("\n=== Testing Model Parameters ===")
    
    model = RNAProteinBindingModel(
        rna_input_dim=4,
        protein_input_dim=1280,
        hidden_dim=256,
        num_attention_heads=8,
        num_layers=3,
        dropout=0.1,
        rna_seq_length=101
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    assert total_params > 0, "Model should have parameters"
    assert trainable_params == total_params, "All parameters should be trainable"
    
    print("✓ Model parameter tests passed!\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Basic Tests for RNA-Protein Binding Prediction")
    print("="*60)
    
    try:
        test_rna_processor()
        test_dataset()
        test_model_forward()
        test_model_parameters()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
