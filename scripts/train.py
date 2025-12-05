"""
Training script for Javanese ASR with LAS-style seq2seq model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
from tqdm import tqdm

from src.model import Seq2SeqASR
from src.dataset import JavaneseASRDataset, collate_fn
from src.vocab import Vocabulary, build_vocab_from_file
from src.features import LogMelFeatureExtractor
from src.metrics import compute_batch_cer
from src.decoder import GreedyDecoder
from src.utils import set_seed, count_parameters, save_checkpoint
from config import Config

from src.data_split import create_speaker_disjoint_split, load_split_info
from pathlib import Path as SplitPath

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    vocab: Vocabulary,
    device: str,
    epoch: int,
    grad_clip_norm: float = 5.0,
) -> float:
    """
    Train for one epoch.
    
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
        
        # Standard FP32 training
        attention_logits, ctc_logits = model(features, feature_lengths, targets, target_lengths)
        
        # Compute encoder lengths (after pyramidal reduction)
        encoder_lengths = feature_lengths // 4  # Approximate reduction factor
        
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
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    decoder: GreedyDecoder,
    vocab: Vocabulary,
    device: str
) -> tuple:
    """
    Validate the model.
    
    Returns:
        (average_loss, average_cer)
    """
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    num_batches = 0
    num_samples = 0
    
    pbar = tqdm(dataloader, desc="Validation")
    for batch in pbar:
        # Move to device
        features = batch['features'].to(device)
        feature_lengths = batch['feature_lengths'].to(device)
        targets = batch['targets'].to(device)
        target_lengths = batch['target_lengths'].to(device)
        transcripts = batch['transcripts']
        
        # Forward pass for loss
        attention_logits, ctc_logits = model(features, feature_lengths, targets, target_lengths)
        
        encoder_lengths = feature_lengths // 4
        
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
        
        # Decode for CER
        hypotheses = decoder.decode(features, feature_lengths)
        
        # Compute CER
        cer = compute_batch_cer(transcripts, hypotheses)
        total_cer += cer * len(transcripts)
        num_samples += len(transcripts)
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'cer': f'{cer:.4f}'})
    
    avg_loss = total_loss / num_batches
    avg_cer = total_cer / num_samples
    
    return avg_loss, avg_cer


def main():
    # Load config
    cfg = Config()
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    
    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build or load vocabulary
    vocab_path = Path(cfg.vocab_path)
    if vocab_path.exists():
        print(f"Loading vocabulary from {vocab_path}")
        vocab = Vocabulary.load(str(vocab_path))
    else:
        print(f"Building vocabulary from {cfg.transcript_file}")
        vocab = build_vocab_from_file(cfg.transcript_file, save_path=str(vocab_path))
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create dataset
    print("Loading dataset...")
    feature_extractor = LogMelFeatureExtractor(
        sample_rate=cfg.sample_rate,
        n_mels=cfg.n_mels,
        win_length_ms=cfg.win_length_ms,
        hop_length_ms=cfg.hop_length_ms
    )
    
    # Create or load split
    if not SplitPath(cfg.split_info_path).exists():
        print("Creating speaker-disjoint split...")
        split_dict = create_speaker_disjoint_split(
            transcript_file=cfg.transcript_file,
            test_speaker_ratio=0.2,  # 20% speakers for test
            val_utterance_ratio=0.1,  # 10% utterances for validation
            seed=cfg.seed,
            save_split_info=True,
            split_info_path=cfg.split_info_path
        )
    else:
        print(f"Loading existing split from {cfg.split_info_path}...")
        split_info = load_split_info(cfg.split_info_path)
        split_dict = split_info['split']
        print(f"  Train: {len(split_dict['train'])} utterances")
        print(f"  Val:   {len(split_dict['val'])} utterances")
        print(f"  Test:  {len(split_dict['test'])} utterances")
    
    # Create datasets with filtered utterance IDs
    print("\nCreating train dataset...")
    train_dataset = JavaneseASRDataset(
        audio_dir=cfg.audio_dir,
        transcript_file=cfg.transcript_file,
        vocab=vocab,
        feature_extractor=feature_extractor,
        apply_cmvn=cfg.apply_cmvn,
        apply_spec_augment=cfg.apply_spec_augment,
        speed_perturb=cfg.speed_perturb,
        utt_id_filter=split_dict['train']
    )
    
    print("\nCreating validation dataset...")
    val_dataset = JavaneseASRDataset(
        audio_dir=cfg.audio_dir,
        transcript_file=cfg.transcript_file,
        vocab=vocab,
        feature_extractor=feature_extractor,
        apply_cmvn=cfg.apply_cmvn,
        apply_spec_augment=False,  # No augmentation for validation
        speed_perturb=False,
        utt_id_filter=split_dict['val']
    )
    
    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(train_dataset)} utterances")
    print(f"  Val:   {len(val_dataset)} utterances")

    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to > 0 for parallel data loading on Linux/Mac
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create model
    print("Creating model...")
    model = Seq2SeqASR(
        vocab_size=len(vocab),
        input_dim=cfg.input_dim,
        encoder_hidden_size=cfg.encoder_hidden_size,
        encoder_num_layers=cfg.encoder_num_layers,
        decoder_dim=cfg.decoder_dim,
        attention_dim=cfg.attention_dim,
        embedding_dim=cfg.embedding_dim,
        dropout=cfg.dropout,
        use_ctc=cfg.use_ctc,
        ctc_weight=cfg.ctc_weight
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    # Create decoder for validation
    greedy_decoder = GreedyDecoder(model, vocab, max_len=cfg.max_decode_len, device=device)
    
    # Create checkpoint directory
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n=== Starting Training for {cfg.num_epochs} epochs ===\n")
    best_val_cer = float('inf')
    
    for epoch in range(1, cfg.num_epochs + 1):
        start_time = time.time()
        
        # Validate
        val_loss, val_cer = validate(model, val_loader, greedy_decoder, vocab, device)
        
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{cfg.num_epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val CER:    {val_cer:.4f}")
        
        # Save checkpoint
        if epoch % cfg.save_every_n_epochs == 0:
            save_checkpoint(
                model, optimizer, epoch, epoch * len(train_loader),
                train_loss, cfg.checkpoint_dir
            )
        
        # Save best model
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            save_checkpoint(
                model, optimizer, epoch, epoch * len(train_loader),
                val_loss, cfg.checkpoint_dir, filename="best_model.pt"
            )
            print(f"  *** New best model (CER: {val_cer:.4f}) ***")
    
    print(f"\n=== Training Complete ===")
    print(f"Best validation CER: {best_val_cer:.4f}")


if __name__ == "__main__":
    main()
