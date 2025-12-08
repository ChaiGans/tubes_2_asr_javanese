"""
Inference script for Javanese ASR
Transcribe audio files using a trained model
"""

import torch
import argparse
from pathlib import Path

from src.model import Seq2SeqASR
from src.vocab import Vocabulary
from src.features import LogMelFeatureExtractor, load_audio, CMVN
from src.decoder import GreedyDecoder
from src.utils import load_checkpoint
from config import Config


def transcribe_audio(
    audio_path: str,
    model: Seq2SeqASR,
    vocab: Vocabulary,
    feature_extractor: LogMelFeatureExtractor,
    decoder: GreedyDecoder,
    device: str = 'cpu'
) -> str:
    """
    Transcribe a single audio file.
    
    Args:
        audio_path: Path to audio file
        model: Trained ASR model
        vocab: Vocabulary
        feature_extractor: Feature extractor
        decoder: Decoder (greedy)
        device: Device to run on
    
    Returns:
        Transcribed text
    """
    # Load audio
    waveform, sr = load_audio(audio_path, target_sr=16000)
    
    # Extract features
    features = feature_extractor(waveform)  # [time, n_mels]
    
    # Apply CMVN
    cmvn = CMVN(norm_means=True, norm_vars=True)
    features = cmvn(features)
    
    # Add batch dimension
    features = features.unsqueeze(0)  # [1, time, n_mels]
    feature_lengths = torch.tensor([features.size(1)], dtype=torch.long)
    
    # Move to device
    features = features.to(device)
    feature_lengths = feature_lengths.to(device)
    
    # Decode
    transcripts = decoder.decode(features, feature_lengths)
    
    return transcripts[0]


def main():
    parser = argparse.ArgumentParser(description="Javanese ASR Inference")
    parser.add_argument('--audio', type=str, required=True, help="Path to audio file or directory")
    parser.add_argument('--checkpoint', type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument('--vocab', type=str, default="vocab.json", help="Path to vocabulary file")
    parser.add_argument('--device', type=str, default="cuda", help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Load config
    cfg = Config()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    print(f"Loading vocabulary from {args.vocab}")
    vocab = Vocabulary.load(args.vocab)
    print(f"Vocabulary size: {len(vocab)}")
    
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
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    
    # Create feature extractor
    feature_extractor = LogMelFeatureExtractor(
        sample_rate=cfg.sample_rate,
        n_mels=cfg.n_mels,
        win_length_ms=cfg.win_length_ms,
        hop_length_ms=cfg.hop_length_ms
    )
    
    decoder = GreedyDecoder(model, vocab, max_len=cfg.max_decode_len, device=device)
    print("Using greedy decoder")

    # Process audio file(s)
    audio_path = Path(args.audio)
    
    if audio_path.is_file():
        # Single file
        print(f"\nTranscribing: {audio_path.name}")
        transcript = transcribe_audio(
            str(audio_path), model, vocab, feature_extractor, decoder, device
        )
        print(f"Transcript: {transcript}")
    
    elif audio_path.is_dir():
        # Directory of files
        audio_files = list(audio_path.glob("*.wav"))
        print(f"\nFound {len(audio_files)} audio files")
        
        for audio_file in audio_files:
            print(f"\nTranscribing: {audio_file.name}")
            transcript = transcribe_audio(
                str(audio_file), model, vocab, feature_extractor, decoder, device
            )
            print(f"Transcript: {transcript}")
    
    else:
        print(f"Error: {audio_path} is not a valid file or directory")


if __name__ == "__main__":
    main()
