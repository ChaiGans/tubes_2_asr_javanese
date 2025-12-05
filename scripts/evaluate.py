"""
Evaluation script for Javanese ASR
Test trained model on closed-set data and compute detailed metrics
"""

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import json

from model import Seq2SeqASR
from vocab import Vocabulary
from features import LogMelFeatureExtractor, load_audio, CMVN
from decoder import GreedyDecoder, BeamSearchDecoder
from metrics import compute_cer, compute_wer
from utils import load_checkpoint
from config import Config


def evaluate_dataset(
    model: Seq2SeqASR,
    vocab: Vocabulary,
    feature_extractor: LogMelFeatureExtractor,
    decoder,
    transcript_file: str,
    audio_dir: str,
    device: str = 'cpu',
    max_samples: int = None
) -> dict:
    """
    Evaluate model on entire dataset.
    
    Returns:
        Dictionary with detailed results
    """
    model.eval()
    
    # Load transcripts
    data = []
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line or (line_idx == 0 and line.startswith('SentenceID')):
                continue
            
            parts = line.split(',')
            if len(parts) >= 2:
                utt_id = parts[0].strip()
                transcript = parts[1].strip()
                
                if not transcript:
                    continue
                
                # Find audio file
                audio_dir_path = Path(audio_dir)
                audio_path = None
                
                # Try different naming conventions
                for pattern in [f"{utt_id}.wav", f"S{utt_id[1:]}.wav", 
                               f"{utt_id.replace('_', '-')}.wav"]:
                    test_path = audio_dir_path / pattern
                    if test_path.exists():
                        audio_path = test_path
                        break
                
                if audio_path:
                    data.append({
                        'utt_id': utt_id,
                        'audio_path': str(audio_path),
                        'reference': transcript
                    })
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"Evaluating on {len(data)} utterances...")
    
    # Evaluate each utterance
    results = []
    total_cer = 0.0
    total_wer = 0.0
    num_errors = 0
    
    cmvn = CMVN(norm_means=True, norm_vars=True)
    
    for item in tqdm(data, desc="Evaluating"):
        try:
            # Load and process audio
            waveform, sr = load_audio(item['audio_path'], target_sr=16000)
            features = feature_extractor(waveform)
            features = cmvn(features)
            
            # Add batch dimension
            features = features.unsqueeze(0).to(device)
            feature_lengths = torch.tensor([features.size(1)], dtype=torch.long).to(device)
            
            # Decode
            hypothesis = decoder.decode(features, feature_lengths)[0]
            
            # Compute metrics
            cer = compute_cer(item['reference'], hypothesis)
            wer = compute_wer(item['reference'], hypothesis)
            
            total_cer += cer
            total_wer += wer
            
            results.append({
                'utt_id': item['utt_id'],
                'reference': item['reference'],
                'hypothesis': hypothesis,
                'cer': cer,
                'wer': wer
            })
            
        except Exception as e:
            print(f"\nError processing {item['utt_id']}: {e}")
            num_errors += 1
            continue
    
    # Compute average metrics
    num_valid = len(results)
    avg_cer = total_cer / num_valid if num_valid > 0 else 0.0
    avg_wer = total_wer / num_valid if num_valid > 0 else 0.0
    
    # Find best and worst examples
    results_sorted_by_cer = sorted(results, key=lambda x: x['cer'])
    best_examples = results_sorted_by_cer[:10]
    worst_examples = results_sorted_by_cer[-10:]
    
    return {
        'num_utterances': num_valid,
        'num_errors': num_errors,
        'avg_cer': avg_cer,
        'avg_wer': avg_wer,
        'all_results': results,
        'best_examples': best_examples,
        'worst_examples': worst_examples
    }


def print_results(results: dict, output_file: str = None):
    """Print evaluation results in a nice format."""
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nDataset Statistics:")
    print(f"  Total utterances: {results['num_utterances']}")
    print(f"  Errors/Skipped: {results['num_errors']}")
    
    print(f"\nOverall Metrics:")
    print(f"  Average CER: {results['avg_cer']*100:.2f}%")
    print(f"  Average WER: {results['avg_wer']*100:.2f}%")
    
    # CER distribution
    all_cers = [r['cer'] for r in results['all_results']]
    print(f"\nCER Distribution:")
    print(f"  Min CER: {min(all_cers)*100:.2f}%")
    print(f"  Max CER: {max(all_cers)*100:.2f}%")
    print(f"  Median CER: {sorted(all_cers)[len(all_cers)//2]*100:.2f}%")
    
    # Perfect predictions
    perfect = sum(1 for r in results['all_results'] if r['cer'] == 0.0)
    print(f"  Perfect predictions (CER=0): {perfect}/{results['num_utterances']} ({perfect/results['num_utterances']*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("BEST 10 PREDICTIONS (Lowest CER)")
    print("="*80)
    for i, ex in enumerate(results['best_examples'], 1):
        print(f"\n{i}. {ex['utt_id']} (CER: {ex['cer']*100:.2f}%)")
        print(f"   REF: {ex['reference']}")
        print(f"   HYP: {ex['hypothesis']}")
    
    print(f"\n{'='*80}")
    print("WORST 10 PREDICTIONS (Highest CER)")
    print("="*80)
    for i, ex in enumerate(results['worst_examples'], 1):
        print(f"\n{i}. {ex['utt_id']} (CER: {ex['cer']*100:.2f}%)")
        print(f"   REF: {ex['reference']}")
        print(f"   HYP: {ex['hypothesis']}")
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Detailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Javanese ASR on closed-set data")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--vocab', type=str, default="vocab.json", help="Path to vocabulary file")
    parser.add_argument('--transcript', type=str, default="transcripts.csv", help="Path to transcript file")
    parser.add_argument('--audio_dir', type=str, default="audio_input", help="Path to audio directory")
    parser.add_argument('--decoder', type=str, default="greedy", choices=['greedy', 'beam'], help="Decoder type")
    parser.add_argument('--beam_size', type=int, default=5, help="Beam size for beam search")
    parser.add_argument('--device', type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument('--max_samples', type=int, default=None, help="Max samples to evaluate (for testing)")
    parser.add_argument('--output', type=str, default="evaluation_results.json", help="Output file for detailed results")
    
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
    
    # Create decoder
    if args.decoder == 'greedy':
        decoder = GreedyDecoder(model, vocab, max_len=cfg.max_decode_len, device=device)
        print("Using greedy decoder")
    else:
        decoder = BeamSearchDecoder(model, vocab, beam_size=args.beam_size, 
                                    max_len=cfg.max_decode_len, device=device)
        print(f"Using beam search decoder (beam size={args.beam_size})")
    
    # Evaluate
    results = evaluate_dataset(
        model=model,
        vocab=vocab,
        feature_extractor=feature_extractor,
        decoder=decoder,
        transcript_file=args.transcript,
        audio_dir=args.audio_dir,
        device=device,
        max_samples=args.max_samples
    )
    
    # Print and save results
    print_results(results, output_file=args.output)


if __name__ == "__main__":
    main()
