"""
Utility functions for Javanese ASR
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List
import random
import numpy as np


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load to
    
    Returns:
        Dictionary with epoch, step, loss
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Step: {checkpoint.get('step', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', float('inf'))
    }


def read_transcript(transcript_file: str) -> List[str]:
    """Read transcripts from CSV file, skipping header."""
    transcripts = []
    with open(transcript_file, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            if count == 0:
                count += 1
                continue
            line = line.strip()
            if not line:
                continue
            # Split by comma (CSV format)
            parts = line.split(',')
            if len(parts) >= 2:
                transcript = parts[1].strip()
                transcripts.append(transcript)
    return transcripts