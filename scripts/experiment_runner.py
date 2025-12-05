"""
Experiment Runner for Javanese ASR (Scratch Model Deep Dive)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import asdict, replace
from typing import Dict, List, Any
import torch
from torch.utils.data import DataLoader
import jiwer
import pandas as pd

from config import Config
from src.model import Seq2SeqASR
from src.dataset import JavaneseASRDataset, collate_fn  
from src.vocab import Vocabulary
from src.features import LogMelFeatureExtractor
from src.decoder import GreedyDecoder, BeamSearchDecoder
from src.utils import set_seed, count_parameters, read_transcript
from scripts.train import train_one_epoch, validate

from src.data_split import create_speaker_disjoint_split, load_split_info
from pathlib import Path as SplitPath


class ExperimentTracker:
    """Track experiments and their results"""
    
    def __init__(self, results_dir: str = "experiment_results_scratch"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results_file = self.results_dir / "all_experiments.json"
        
        if self.results_file.exists():
            with open(self.results_file, 'r', encoding='utf-8') as f:
                self.all_results = json.load(f)
        else:
            self.all_results = []
    
    def save_experiment(self, experiment_config: Dict[str, Any], results: Dict[str, Any]):
        experiment_data = {
            "experiment_id": len(self.all_results) + 1,
            "timestamp": datetime.now().isoformat(),
            "config": experiment_config,
            "results": results
        }
        
        self.all_results.append(experiment_data)
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
        
        exp_file = self.results_dir / f"experiment_{experiment_data['experiment_id']:03d}.json"
        with open(exp_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False)
        
        return experiment_data['experiment_id']
    
    def get_summary(self) -> str:
        if not self.all_results:
            return "No experiments run yet."
        
        summary = "\n" + "="*120 + "\n"
        summary += "EXPERIMENT SUMMARY (Scratch Deep Dive)\n"
        summary += "="*120 + "\n\n"
        
        sorted_results = sorted(
            self.all_results, 
            key=lambda x: x['results'].get('best_val_wer', float('inf'))
        )
        
        summary += f"{'ID':<4} {'Vocab':<6} {'Enc':<10} {'Dec':<6} {'LR':<8} {'WER':<8} {'CER':<8} {'Time(m)':<8}\n"
        summary += "-"*120 + "\n"
        
        for exp in sorted_results:
            exp_id = exp['experiment_id']
            cfg = exp['config']
            res = exp['results']
            
            vocab = cfg.get('token_type', 'char')
            enc = cfg.get('encoder_type', 'pyramidal')
            dec = cfg.get('decoder_type', 'lstm')
            lr = f"{cfg.get('learning_rate', 0):.0e}"
            wer = f"{res.get('best_val_wer', 999):.4f}"
            cer = f"{res.get('final_val_cer', 999):.4f}"
            time_m = f"{res.get('training_time_seconds', 0)/60:.1f}"
            
            summary += f"{exp_id:<4} {vocab:<6} {enc:<10} {dec:<6} {lr:<8} {wer:<8} {cer:<8} {time_m:<8}\n"
        
        return summary


class ExperimentRunner:
    def __init__(self, base_config: Config, tracker: ExperimentTracker):
        self.base_config = base_config
        self.tracker = tracker
        
    def prepare_data(self, config: Config, token_type: str):
        # Load vocabulary with specific token type
        # Note: We rebuild it here to ensure it matches the requested type
        # In a real scenario, we might want to cache these
        print(f"Building {token_type}-level vocabulary...")
        
        # Use utility to read transcripts correctly (handles CSV format and header)
        transcripts = read_transcript(config.transcript_file)
        
        vocab = Vocabulary(token_type=token_type)
        vocab.build_from_transcripts(transcripts, min_freq=1)
        print(f"Vocabulary size: {len(vocab)}")
        
        feature_extractor = LogMelFeatureExtractor(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels
        )
        
        if not SplitPath(config.split_info_path).exists():
            print("Creating speaker-disjoint split...")
            split_dict = create_speaker_disjoint_split(
                transcript_file=config.transcript_file,
                test_speaker_ratio=0.2,
                val_utterance_ratio=0.1,
                seed=config.seed,
                save_split_info=True,
                split_info_path=config.split_info_path
            )
        else:
            split_info = load_split_info(config.split_info_path)
            split_dict = split_info['split']
        
        train_dataset = JavaneseASRDataset(
            audio_dir=config.audio_dir,
            transcript_file=config.transcript_file,
            vocab=vocab,
            feature_extractor=feature_extractor,
            apply_spec_augment=config.apply_spec_augment,
            utt_id_filter=split_dict['train']
        )
        
        val_dataset = JavaneseASRDataset(
            audio_dir=config.audio_dir,
            transcript_file=config.transcript_file,
            vocab=vocab,
            feature_extractor=feature_extractor,
            apply_spec_augment=False,
            utt_id_filter=split_dict['val']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        return vocab, train_loader, val_loader
    
    def run_single_experiment(self, experiment_name: str, config_overrides: Dict[str, Any]):
        print("\n" + "="*80)
        print(f"🔬 EXPERIMENT: {experiment_name}")
        print("="*80)
        
        current_config = replace(self.base_config, **config_overrides)
        token_type = config_overrides.get("token_type", "char")
        encoder_type = config_overrides.get("encoder_type", "pyramidal")
        decoder_type = config_overrides.get("decoder_type", "lstm")
        
        set_seed(current_config.seed)
        
        print("\n📋 Configuration:")
        for key, value in config_overrides.items():
            print(f"   {key}: {value}")
        
        vocab, train_loader, val_loader = self.prepare_data(current_config, token_type)
        
        print(f"\n🏗️  Building model...")
        model = Seq2SeqASR(
            vocab_size=len(vocab),
            input_dim=current_config.input_dim,
            encoder_hidden_size=current_config.encoder_hidden_size,
            encoder_num_layers=current_config.encoder_num_layers,
            decoder_dim=current_config.decoder_dim,
            attention_dim=current_config.attention_dim,
            embedding_dim=current_config.embedding_dim,
            dropout=current_config.dropout,
            use_ctc=current_config.use_ctc,
            ctc_weight=current_config.ctc_weight,
            encoder_type=encoder_type,
            decoder_type=decoder_type
        ).to(current_config.device)
        
        print(f"   Model parameters: {count_parameters(model):,}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=current_config.learning_rate)
        
        print(f"\n🏃 Training for {current_config.num_epochs} epochs...")
        start_time = time.time()
        
        train_losses = []
        val_cers = []
        val_wers = []
        best_val_wer = float('inf')
        
        for epoch in range(1, current_config.num_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, vocab, current_config.device, epoch, current_config.grad_clip_norm
            )
            train_losses.append(train_loss)
            
            print(f"\n   Validating epoch {epoch}...")
            
            # Use Greedy Decoder for validation speed
            decoder = GreedyDecoder(model, vocab, max_len=current_config.max_decode_len, device=current_config.device)
            
            val_loss, val_cer, predictions, references = self.validate_with_metrics(
                model, val_loader, decoder, vocab, current_config.device, encoder_type
            )
            
            val_wer = jiwer.wer(references, predictions)
            val_cers.append(val_cer)
            val_wers.append(val_wer)
            
            print(f"   Epoch {epoch}: Loss={train_loss:.4f}, CER={val_cer:.4f}, WER={val_wer:.4f}")
            
            if val_wer < best_val_wer:
                best_val_wer = val_wer
                print(f"   ⭐ New best WER: {best_val_wer:.4f}")
                
                checkpoint_dir = Path("experiment_checkpoints_scratch")
                checkpoint_dir.mkdir(exist_ok=True)
                safe_name = experiment_name.replace(" ", "_").replace(":", "")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_wer': best_val_wer,
                    'config': config_overrides
                }, checkpoint_dir / f"{safe_name}_best.pt")
        
        training_time = time.time() - start_time
        
        results = {
            "experiment_name": experiment_name,
            "train_losses": train_losses,
            "val_cers": val_cers,
            "val_wers": val_wers,
            "best_val_wer": best_val_wer,
            "final_val_cer": val_cers[-1],
            "training_time_seconds": training_time
        }
        
        self.tracker.save_experiment(config_overrides, results)
        return results

    def validate_with_metrics(self, model, dataloader, decoder, vocab, device, encoder_type):
        model.eval()
        total_loss = 0
        all_preds = []
        all_refs = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(device)
                feature_lengths = batch['feature_lengths'].to(device)
                targets = batch['targets'].to(device)
                target_lengths = batch['target_lengths'].to(device)
                
                attention_logits, ctc_logits = model(features, feature_lengths, targets, target_lengths)
                
                # Determine encoder length reduction
                enc_len = feature_lengths // 4 if encoder_type == "pyramidal" else feature_lengths
                
                loss = model.compute_loss(
                    attention_logits, targets, target_lengths, ctc_logits, enc_len, vocab.pad_idx, vocab.blank_idx
                )
                total_loss += loss.item()
                
                # Decode
                decoded_indices, _ = decoder.decode_batch(
                    model.encoder(features, feature_lengths)[0], enc_len
                )
                
                for i in range(len(targets)):
                    pred = vocab.decode(decoded_indices[i], remove_special=True)
                    ref = vocab.decode(targets[i].tolist(), remove_special=True)
                    all_preds.append(pred)
                    all_refs.append(ref)
        
        avg_loss = total_loss / len(dataloader)
        cer = jiwer.cer(all_refs, all_preds)
        
        return avg_loss, cer, all_preds, all_refs


def load_experiments_from_json(json_path: str) -> List[Dict[str, Any]]:
    """Load experiment definitions from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📋 Loaded experiment config: {data.get('name', 'Unknown')}")
    if 'description' in data:
        print(f"   {data['description']}")
    
    return data.get('experiments', [])


def config_matches(config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
    """Check if two experiment configs are equivalent."""
    # Compare key config parameters
    keys_to_compare = ['token_type', 'encoder_type', 'decoder_type', 'learning_rate', 'num_epochs']
    for key in keys_to_compare:
        if config1.get(key) != config2.get(key):
            return False
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ASR experiments from JSON config')
    parser.add_argument('--config', '-c', type=str, default='experiments/scratch_experiments_v1.json',
                        help='Path to experiment configuration JSON file')
    parser.add_argument('--skip-existing', '-s', action='store_true',
                        help='Skip experiments that have already been run with same config')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List experiments without running them')
    parser.add_argument('--only', '-o', type=str, nargs='+',
                        help='Only run experiments with names containing these strings (e.g., --only S01 S02)')
    
    args = parser.parse_args()
    
    print("🔬 Javanese ASR Scratch Model Deep Dive")
    print("="*80)
    
    # Load experiments from JSON
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {args.config}")
        return
    
    experiments = load_experiments_from_json(args.config)
    
    if not experiments:
        print("❌ No experiments found in config file")
        return
    
    # Filter experiments if --only is specified
    if args.only:
        filtered = []
        for exp in experiments:
            for pattern in args.only:
                if pattern.lower() in exp['name'].lower():
                    filtered.append(exp)
                    break
        experiments = filtered
        print(f"📌 Filtered to {len(experiments)} experiments matching: {args.only}")
    
    # List mode
    if args.list:
        print(f"\n📋 Experiments in {args.config}:")
        print("-" * 60)
        for i, exp in enumerate(experiments, 1):
            cfg = exp['config']
            print(f"{i}. {exp['name']}")
            print(f"   token={cfg.get('token_type')}, encoder={cfg.get('encoder_type')}, "
                  f"decoder={cfg.get('decoder_type')}, lr={cfg.get('learning_rate')}, "
                  f"epochs={cfg.get('num_epochs')}")
        return
    
    base_config = Config()
    tracker = ExperimentTracker()
    runner = ExperimentRunner(base_config, tracker)
    
    # Track which experiments to skip
    experiments_to_run = []
    if args.skip_existing and tracker.all_results:
        for exp in experiments:
            already_run = False
            for prev_exp in tracker.all_results:
                if config_matches(exp['config'], prev_exp['config']):
                    print(f"⏭️  Skipping {exp['name']} (already run as experiment {prev_exp['experiment_id']})")
                    already_run = True
                    break
            if not already_run:
                experiments_to_run.append(exp)
    else:
        experiments_to_run = experiments
    
    if not experiments_to_run:
        print("\n✅ All experiments have already been run!")
        print(tracker.get_summary())
        return
    
    print(f"\n🚀 Running {len(experiments_to_run)} experiments...")
    
    for i, exp in enumerate(experiments_to_run, 1):
        print(f"\nRUNNING {i}/{len(experiments_to_run)}")
        try:
            runner.run_single_experiment(exp["name"], exp["config"])
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "="*80)
    print(tracker.get_summary())
    with open(tracker.results_dir / "summary.txt", 'w') as f:
        f.write(tracker.get_summary())

if __name__ == "__main__":
    main()
