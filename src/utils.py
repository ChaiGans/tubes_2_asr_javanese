"""
Utility functions for Javanese ASR
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: str,
    filename: str = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current training step
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoint
        filename: Optional custom filename
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch{epoch}_step{step}.pt"
    
    checkpoint_path = checkpoint_dir / filename
    
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")


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


def create_mask(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """
    Create mask from lengths.
    
    Args:
        lengths: [batch] - sequence lengths
        max_len: Maximum length (if None, use max of lengths)
    
    Returns:
        mask: [batch, max_len] - 1 for valid positions, 0 for padding
    """
    batch_size = lengths.size(0)
    if max_len is None:
        max_len = lengths.max().item()
    
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    mask = (mask < lengths.unsqueeze(1)).long()
    
    return mask


if __name__ == "__main__":
    import tempfile
    
    # Test utilities
    print("Testing utilities...")
    
    # Test seed setting
    set_seed(42)
    
    # Test parameter counting
    dummy_model = nn.Linear(10, 5)
    num_params = count_parameters(dummy_model)
    print(f"Dummy model parameters: {num_params}")
    
    # Test checkpoint save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        optimizer = torch.optim.Adam(dummy_model.parameters())
        
        # Save
        save_checkpoint(dummy_model, optimizer, epoch=1, step=100, loss=0.5, 
                       checkpoint_dir=tmpdir, filename="test.pt")
        
        # Load
        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        info = load_checkpoint(Path(tmpdir) / "test.pt", new_model, new_optimizer)
        
        print(f"Loaded info: {info}")
    
    # Test mask creation
    lengths = torch.tensor([5, 3, 7])
    mask = create_mask(lengths, max_len=10)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask:\n{mask}")
    
    print("Utility tests passed!")
