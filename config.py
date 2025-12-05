"""
Configuration for Javanese ASR Training
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for model training and architecture."""
    
    # Data paths
    audio_dir: str = "data/audio_input"
    transcript_file: str = "data/transcripts.csv"
    vocab_path: str = "data/vocab.json"
    
    # Training - 🚀 OPTIMIZED FOR GPU
    batch_size: int = 64  # 🚀 Increased from 8 (8x more GPU work!)
    num_epochs: int = 2
    learning_rate: float = 1e-3
    grad_clip_norm: float = 5.0
    teacher_forcing_ratio: float = 1.0
    
    # Model architecture - 🚀 LARGER FOR MORE GPU UTILIZATION
    input_dim: int = 80  # Log-mel features
    encoder_hidden_size: int = 256  # 🚀 Increased from 128 (2x)
    encoder_num_layers: int = 3
    decoder_dim: int = 512  # 🚀 Increased from 256 (2x)
    attention_dim: int = 256  # 🚀 Increased from 128 (2x)
    embedding_dim: int = 128  # 🚀 Increased from 64 (2x)
    dropout: float = 0.3
    
    # CTC settings
    use_ctc: bool =  False  # Set to True for joint CTC+attention
    ctc_weight: float = 0.3  # Weight for CTC loss (lambda)
    
    # Feature extraction
    sample_rate: int = 16000
    n_mels: int = 80
    win_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    
    # Augmentation
    apply_cmvn: bool = True
    apply_spec_augment: bool = True
    speed_perturb: bool = False  # Set to True to enable speed perturbation
    
    # Validation
    val_split: float = 0.1  # 10% for validation
    val_every_n_steps: int = 500
    
    # Checkpointing
    checkpoint_dir: str = "results/checkpoints"
    save_every_n_epochs: int = 1
    
    # Decoding
    max_decode_len: int = 200
    beam_size: int = 5
    
    # Device
    device: str = "cuda"  # or "cpu"
    
    # Random seed
    seed: int = 42
    
    # 🚀 GPU Optimization Parameters (NEW!)
    num_workers: int = 6  # Parallel data loading (was 0)
    pin_memory: bool = True  # Faster CPU→GPU transfers
    prefetch_factor: int = 2  # Prefetch batches ahead
    persistent_workers: bool = True  # Reuse worker processes

if __name__ == "__main__":
    cfg = Config()
    print("Default configuration:")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Encoder layers: {cfg.encoder_num_layers}")
    print(f"  Use CTC: {cfg.use_ctc}")
