"""
Training script for Javanese ASR with LAS-style seq2seq model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.vocab import Vocabulary
from src.metrics import compute_batch_cer
import jiwer

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    vocab: Vocabulary,
    device: str,
    epoch: int,
    grad_clip_norm: float = 5.0,
    encoder_type: str = "pyramidal"
) -> float:
    """
    Train for one epoch.
    
    Args:
        encoder_type: "pyramidal" or "standard" (affects encoder length calculation)
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        # Move to device
        features = batch['features'].to(device)
        feature_lengths = batch['feature_lengths'].to(device)
        targets = batch['targets'].to(device)
        target_lengths = batch['target_lengths'].to(device)
        
        # Scheduled Sampling: gradually reduce teacher forcing
        # Start at 1.0, decay to 0.5 over ~50 epochs
        # This forces the model to learn from its own predictions
        tf_ratio = max(0.5, 1.0 - (epoch * 0.01))
        
        # Forward pass with scheduled teacher forcing
        attention_logits, ctc_logits = model(features, feature_lengths, targets, teacher_forcing_ratio=tf_ratio)
        
        # Compute encoder lengths based on encoder type
        # Pyramidal: 2 levels of 2x reduction = 4x total reduction
        # Standard: no reduction
        encoder_lengths = feature_lengths // 4 if encoder_type == "pyramidal" else feature_lengths
        
        # Compute loss
        loss = model.compute_loss(
            attention_logits=attention_logits,
            targets=targets,
            target_lengths=target_lengths,
            ctc_logits=ctc_logits,
            encoder_lengths=encoder_lengths,
            pad_idx=vocab.pad_idx,
            blank_idx=vocab.blank_idx
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

@torch.no_grad()
def validate_with_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    decoder,
    vocab,
    device: str,
    encoder_type: str = "pyramidal"
) -> tuple:
    """
    Validate the model and return predictions/references for WER calculation.
    
    Args:
        model: The ASR model
        dataloader: Validation dataloader
        decoder: Decoder instance (GreedyDecoder)
        vocab: Vocabulary instance
        device: Device string
        encoder_type: "pyramidal" or "standard" (affects encoder length calculation)
    
    Returns:
        (average_loss, average_cer, average_wer, predictions, references)
    """
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    num_batches = 0
    num_samples = 0
    all_predictions = []
    all_references = []
    
    for batch in dataloader:
        features = batch['features'].to(device)
        feature_lengths = batch['feature_lengths'].to(device)
        targets = batch['targets'].to(device)
        target_lengths = batch['target_lengths'].to(device)
        transcripts = batch['transcripts']
        
        # Forward pass (NO teacher forcing during validation!)
        attention_logits, ctc_logits = model(features, feature_lengths, targets, teacher_forcing_ratio=0.0)
        
        # Determine encoder length reduction
        encoder_lengths = feature_lengths // 4 if encoder_type == "pyramidal" else feature_lengths
        
        loss = model.compute_loss(
            attention_logits=attention_logits,
            targets=targets,
            target_lengths=target_lengths,
            ctc_logits=ctc_logits,
            encoder_lengths=encoder_lengths,
            pad_idx=vocab.pad_idx,
            blank_idx=vocab.blank_idx
        )
        
        total_loss += loss.item()
        
        # Decode predictions
        hypotheses = decoder.decode(features, feature_lengths)
        
        # Compute CER and collect predictions/references
        cer = compute_batch_cer(transcripts, hypotheses)
        total_cer += cer * len(transcripts)
        num_samples += len(transcripts)
        num_batches += 1
        
        all_predictions.extend(hypotheses)
        all_references.extend(transcripts)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_cer = total_cer / num_samples if num_samples > 0 else 0.0
    
    # Compute WER using jiwer
    avg_wer = jiwer.wer(all_references, all_predictions) if all_references else 0.0
    
    return avg_loss, avg_cer, avg_wer, all_predictions, all_references