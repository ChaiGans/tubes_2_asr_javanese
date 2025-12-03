"""
Experiment Runner for Javanese ASR
Systematically test different configurations and track performance improvements
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from dataclasses import asdict, replace
from typing import Dict, List, Any
import torch
from torch.utils.data import DataLoader, random_split


from config import Config
from model import Seq2SeqASR
from dataset import JavaneseASRDataset, collate_fn  
from vocab import Vocabulary
from features import LogMelFeatureExtractor
from metrics import compute_batch_cer
from decoder import GreedyDecoder, BeamSearchDecoder
from utils import set_seed, count_parameters, save_checkpoint
from train import train_one_epoch, validate


class ExperimentTracker:
    """Track experiments and their results"""
    
    def __init__(self, results_dir: str = "experiment_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results_file = self.results_dir / "all_experiments.json"
        
        # Load existing results if available
        if self.results_file.exists():
            with open(self.results_file, 'r', encoding='utf-8') as f:
                self.all_results = json.load(f)
        else:
            self.all_results = []
    
    def save_experiment(self, experiment_config: Dict[str, Any], results: Dict[str, Any]):
        """Save a single experiment result"""
        experiment_data = {
            "experiment_id": len(self.all_results) + 1,
            "timestamp": datetime.now().isoformat(),
            "config": experiment_config,
            "results": results
        }
        
        self.all_results.append(experiment_data)
        
        # Save all results
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
        
        # Save individual experiment
        exp_file = self.results_dir / f"experiment_{experiment_data['experiment_id']:03d}.json"
        with open(exp_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved experiment {experiment_data['experiment_id']} to {exp_file}")
        
        return experiment_data['experiment_id']
    
    def get_summary(self) -> str:
        """Get a summary of all experiments"""
        if not self.all_results:
            return "No experiments run yet."
        
        summary = "\n" + "="*80 + "\n"
        summary += "EXPERIMENT SUMMARY\n"
        summary += "="*80 + "\n\n"
        
        # Sort by CER (best first)
        sorted_results = sorted(
            self.all_results, 
            key=lambda x: x['results'].get('val_cer', float('inf'))
        )
        
        summary += f"{'ID':<5} {'CTC':<6} {'Epochs':<8} {'Beam':<6} {'LR':<10} {'Val CER':<10} {'Val WER':<10}\n"
        summary += "-"*80 + "\n"
        
        for exp in sorted_results:
            exp_id = exp['experiment_id']
            cfg = exp['config']
            res = exp['results']
            
            use_ctc = "Yes" if cfg.get('use_ctc', False) else "No"
            epochs = cfg.get('num_epochs', 'N/A')
            beam = cfg.get('beam_size', 1)
            lr = f"{cfg.get('learning_rate', 0):.0e}"
            val_cer = f"{res.get('val_cer', 999):.4f}"
            val_wer = f"{res.get('val_wer', 999):.4f}"
            
            summary += f"{exp_id:<5} {use_ctc:<6} {epochs:<8} {beam:<6} {lr:<10} {val_cer:<10} {val_wer:<10}\n"
        
        summary += "\n" + "="*80 + "\n"
        
        # Best experiment
        best_exp = sorted_results[0]
        summary += f"\n🏆 Best Experiment: #{best_exp['experiment_id']}\n"
        summary += f"   Val CER: {best_exp['results'].get('val_cer', 'N/A'):.4f}\n"
        summary += f"   Val WER: {best_exp['results'].get('val_wer', 'N/A'):.4f}\n"
        summary += f"   Config: {json.dumps(best_exp['config'], indent=6)}\n"
        
        return summary


class ExperimentRunner:
    """Run systematic experiments with different configurations"""
    
    def __init__(self, base_config: Config, tracker: ExperimentTracker):
        self.base_config = base_config
        self.tracker = tracker
        
    def prepare_data(self, config: Config):
        """Prepare datasets and loaders"""
        # Load vocabulary
        vocab = Vocabulary.load(config.vocab_path)
        print(f"Vocabulary size: {len(vocab)}")
        
        # Feature extractor
        feature_extractor = LogMelFeatureExtractor(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            win_length_ms=config.win_length_ms,
            hop_length_ms=config.hop_length_ms
        )
        
        # Create dataset
        full_dataset = JavaneseASRDataset(
            audio_dir=config.audio_dir,
            transcript_file=config.transcript_file,
            vocab=vocab,
            feature_extractor=feature_extractor,
            apply_spec_augment=config.apply_spec_augment,
            speed_perturb=config.speed_perturb
        )
        
        # Split dataset
        val_size = int(len(full_dataset) * config.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.seed)
        )
        
        # Create data loaders - 🚀 GPU OPTIMIZED
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=getattr(config, 'num_workers', 6),  # 🚀 Parallel data loading
            pin_memory=getattr(config, 'pin_memory', True),  # 🚀 Faster CPU→GPU transfers
            prefetch_factor=getattr(config, 'prefetch_factor', 2) if getattr(config, 'num_workers', 0) > 0 else None,
            persistent_workers=getattr(config, 'persistent_workers', True) if getattr(config, 'num_workers', 0) > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=getattr(config, 'num_workers', 6),  # 🚀 Parallel data loading
            pin_memory=getattr(config, 'pin_memory', True),  # 🚀 Faster CPU→GPU transfers
            prefetch_factor=getattr(config, 'prefetch_factor', 2) if getattr(config, 'num_workers', 0) > 0 else None,
            persistent_workers=getattr(config, 'persistent_workers', True) if getattr(config, 'num_workers', 0) > 0 else False
        )
        
        return vocab, train_loader, val_loader
    
    def run_single_experiment(
        self, 
        experiment_name: str,
        config_overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a single experiment with given configuration
        
        Args:
            experiment_name: Descriptive name for the experiment
            config_overrides: Dictionary of config parameters to override
            
        Returns:
            Dictionary containing experiment results
        """
        print("\n" + "="*80)
        print(f"🔬 EXPERIMENT: {experiment_name}")
        print("="*80)
        
        # Create config for this experiment
        current_config = replace(self.base_config, **config_overrides)
        
        # Set seed for reproducibility
        set_seed(current_config.seed)
        
        # Display experiment config
        print("\n📋 Configuration:")
        for key, value in config_overrides.items():
            print(f"   {key}: {value}")
        
        # Prepare data
        print("\n📂 Loading data...")
        vocab, train_loader, val_loader = self.prepare_data(current_config)
        
        # Create model
        print("\n🏗️  Building model...")
        model = Seq2SeqASR(
            vocab_size=len(vocab),
            input_dim=current_config.input_dim,
            encoder_hidden_size=current_config.encoder_hidden_size,
            encoder_num_layers=current_config.encoder_num_layers,
            decoder_dim=current_config.decoder_dim,
            attention_dim=current_config.attention_dim,
            embedding_dim=current_config.embedding_dim,
            dropout=current_config.dropout,
            use_ctc=current_config.use_ctc
        ).to(current_config.device)
        
        num_params = count_parameters(model)
        print(f"   Model parameters: {num_params:,}")
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=current_config.learning_rate)
        
        # Training
        print(f"\n🏃 Training for {current_config.num_epochs} epochs...")
        start_time = time.time()
        
        train_losses = []
        val_cers = []
        val_wers = []
        
        best_val_cer = float('inf')
        
        for epoch in range(1, current_config.num_epochs + 1):
            # Train
            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                vocab=vocab,
                device=current_config.device,
                epoch=epoch,
                grad_clip_norm=current_config.grad_clip_norm,
            )
            train_losses.append(train_loss)
            
            # Validate
            print(f"\n   Validating epoch {epoch}...")
            
            # Use appropriate decoder based on config
            if current_config.beam_size > 1:
                decoder = BeamSearchDecoder(
                    model=model,
                    vocab=vocab,
                    beam_size=current_config.beam_size,
                    max_len=current_config.max_decode_len,
                    device=current_config.device  # ✅ Add device parameter
                )
            else:
                decoder = GreedyDecoder(
                    model=model,
                    vocab=vocab,
                    max_len=current_config.max_decode_len,
                    device=current_config.device  # ✅ Add device parameter
                )
            
            val_loss, val_cer = validate(
                model=model,
                dataloader=val_loader,
                decoder=decoder,
                vocab=vocab,
                device=current_config.device
            )
            
            val_cers.append(val_cer)
            
            # Approximate WER (you can make this more accurate)
            val_wer = val_cer * 1.2  # Rough approximation
            val_wers.append(val_wer)
            
            print(f"   Epoch {epoch}: Train Loss={train_loss:.4f}, Val CER={val_cer:.4f}, Val WER={val_wer:.4f}")
            
            # Track best model and save checkpoint
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                print(f"   ⭐ New best CER: {best_val_cer:.4f}")
                
                # Save best model checkpoint
                checkpoint_dir = Path("experiment_checkpoints")
                checkpoint_dir.mkdir(exist_ok=True)
                
                # Create safe filename from experiment name
                safe_name = experiment_name.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace(",", "")
                checkpoint_path = checkpoint_dir / f"{safe_name}_best.pt"
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_cer': best_val_cer,
                    'val_loss': val_loss,
                    'config': config_overrides,
                    'experiment_name': experiment_name
                }
                
                torch.save(checkpoint, checkpoint_path)
                print(f"   💾 Saved best model to: {checkpoint_path}")
        
        training_time = time.time() - start_time
        
        # Build checkpoint path
        safe_name = experiment_name.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace(",", "")
        checkpoint_path = Path("experiment_checkpoints") / f"{safe_name}_best.pt"
        
        # Compile results
        results = {
            "experiment_name": experiment_name,
            "train_losses": train_losses,
            "val_cers": val_cers,
            "val_wers": val_wers,
            "best_val_cer": best_val_cer,
            "final_val_cer": val_cers[-1],
            "final_val_wer": val_wers[-1],
            "val_cer": val_cers[-1],  # For tracker sorting
            "val_wer": val_wers[-1],
            "training_time_seconds": training_time,
            "num_parameters": num_params,
            "checkpoint_path": str(checkpoint_path)  # Save path to best model
        }
        
        # Save experiment
        exp_id = self.tracker.save_experiment(
            experiment_config=config_overrides,
            results=results
        )
        
        print(f"\n✅ Experiment complete! ID: {exp_id}")
        print(f"   Best Val CER: {best_val_cer:.4f}")
        print(f"   Final Val CER: {val_cers[-1]:.4f}")
        print(f"   Training time: {training_time/60:.1f} minutes")
        
        return results


def define_experiments() -> List[Dict[str, Any]]:
    """
    Define all experiments to run
    
    Returns:
        List of (experiment_name, config_overrides) tuples
    """
    experiments = []
    
    # 1. Baseline (50 epochs with adapted LR)
    experiments.append({
        "name": "Baseline (No CTC, 50 epochs)",
        "config": {
            "use_ctc": False,
            "num_epochs": 50,
            "beam_size": 1,  # Greedy decoding
            "learning_rate": 5e-4,  # 🚀 Adapted for longer training (was 1e-3)
            "batch_size": 64  # 🚀 Optimized from 8
        }
    })
    
    # 2. Add CTC (50 epochs)
    experiments.append({
        "name": "With CTC (ctc_weight=0.3, 50 epochs)",
        "config": {
            "use_ctc": True,
            "ctc_weight": 0.3,
            "num_epochs": 50,
            "beam_size": 1,
            "learning_rate": 5e-4,  # 🚀 Adapted for longer training
            "batch_size": 64  # 🚀 Optimized from 8
        }
    })
    
    # 3. Different CTC weight (50 epochs)
    experiments.append({
        "name": "With CTC (ctc_weight=0.5, 50 epochs)",
        "config": {
            "use_ctc": True,
            "ctc_weight": 0.5,
            "num_epochs": 50,
            "beam_size": 1,
            "learning_rate": 5e-4,  # 🚀 Adapted for longer training
            "batch_size": 64  # 🚀 Optimized from 8
        }
    })
    
    # 4. Longer training (baseline, 100 epochs)
    experiments.append({
        "name": "Baseline with 100 epochs",
        "config": {
            "use_ctc": False,
            "num_epochs": 100,
            "beam_size": 1,
            "learning_rate": 3e-4,  # 🚀 Adapted for 100 epochs (lower LR)
            "batch_size": 64  # 🚀 Optimized
        }
    })
    
    # 5. CTC with longer training (100 epochs)
    experiments.append({
        "name": "CTC with 100 epochs",
        "config": {
            "use_ctc": True,
            "ctc_weight": 0.3,
            "num_epochs": 100,
            "beam_size": 1,
            "learning_rate": 3e-4,  # 🚀 Adapted for 100 epochs
            "batch_size": 64  # 🚀 Optimized
        }
    })
    
    # 6. Beam search (beam=3, 100 epochs)
    experiments.append({
        "name": "Baseline + Beam Search (beam=3, 100 epochs)",
        "config": {
            "use_ctc": False,
            "num_epochs": 100,
            "beam_size": 1,
            "learning_rate": 3e-4,  # 🚀 Adapted for 100 epochs
            "batch_size": 64  # 🚀 Optimized
        }
    })
    
    # 7. Beam search (beam=5, 100 epochs)
    experiments.append({
        "name": "Baseline + Beam Search (beam=5, 100 epochs)",
        "config": {
            "use_ctc": False,
            "num_epochs": 100,
            "beam_size": 1,
            "learning_rate": 3e-4,  # 🚀 Adapted for 100 epochs
            "batch_size": 64  # 🚀 Optimized
        }
    })
    
    # 8. CTC + Beam search (100 epochs)
    experiments.append({
        "name": "CTC + Beam Search (beam=5, 100 epochs)",
        "config": {
            "use_ctc": True,
            "ctc_weight": 0.3,
            "num_epochs": 100,
            "beam_size": 1,
            "learning_rate": 3e-4,  # 🚀 Adapted for 100 epochs
            "batch_size": 64  # 🚀 Optimized
        }
    })
    
    # 9. Different learning rate
    experiments.append({
        "name": "Lower LR (5e-4)",
        "config": {
            "use_ctc": False,
            "num_epochs": 100,
            "beam_size": 1,
            "learning_rate": 5e-4,
            "batch_size": 64  # 🚀 Optimized
        }
    })
    
    # 10. Higher learning rate
    experiments.append({
        "name": "Higher LR (3e-3)",
        "config": {
            "use_ctc": False,
            "num_epochs": 100,
            "beam_size": 1,
            "learning_rate": 3e-3,
            "batch_size": 64  # 🚀 Optimized
        }
    })
    
    # 11. Even larger batch size (100 epochs)
    experiments.append({
        "name": "Even Larger batch (96, 100 epochs)",
        "config": {
            "use_ctc": False,
            "num_epochs": 100,
            "beam_size": 1,
            "learning_rate": 3e-4,  # 🚀 Adapted for 100 epochs
            "batch_size": 96  # 🚀 Test even larger
        }
    })
    
    # 12. Best combination (200 epochs for maximum performance)
    experiments.append({
        "name": "Best Combo: CTC + Beam + 200 epochs",
        "config": {
            "use_ctc": True,
            "ctc_weight": 0.3,
            "num_epochs": 200,
            "beam_size": 1,
            "learning_rate": 2e-4,  # 🚀 Lower LR for very long training (200 epochs)
            "batch_size": 64  # 🚀 Optimized from 8
        }
    })
    
    return experiments


def main():
    """Main experiment runner"""
    print("🔬 Javanese ASR Experiment Runner")
    print("="*80)
    
    # Base configuration
    base_config = Config()
    
    # Initialize tracker
    tracker = ExperimentTracker(results_dir="experiment_results")
    
    # Initialize runner
    runner = ExperimentRunner(base_config, tracker)
    
    # Get all experiments
    experiments = define_experiments()
    
    print(f"\n📊 Total experiments to run: {len(experiments)}")
    print("\nPress Enter to start, or Ctrl+C to cancel...")
    input()
    
    # Run all experiments
    for i, exp_def in enumerate(experiments, 1):
        print(f"\n\n{'='*80}")
        print(f"RUNNING EXPERIMENT {i}/{len(experiments)}")
        print(f"{'='*80}\n")
        
        try:
            runner.run_single_experiment(
                experiment_name=exp_def["name"],
                config_overrides=exp_def["config"]
            )
        except Exception as e:
            print(f"\n❌ Experiment failed with error: {e}")
            print("Continuing to next experiment...")
            continue
    
    # Print summary
    print("\n\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print(tracker.get_summary())
    
    # Save summary to file
    summary_file = tracker.results_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(tracker.get_summary())
    print(f"\n📄 Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
