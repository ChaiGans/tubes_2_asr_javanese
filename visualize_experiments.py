"""
Visualize and analyze experiment results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


class ExperimentVisualizer:
    """Visualize experiment results"""
    
    def __init__(self, results_dir: str = "experiment_results"):
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / "all_experiments.json"
        
        if not self.results_file.exists():
            raise FileNotFoundError(f"No experiments found at {self.results_file}")
        
        # Load all experiments
        with open(self.results_file, 'r', encoding='utf-8') as f:
            self.all_results = json.load(f)
        
        print(f"Loaded {len(self.all_results)} experiments")
    
    def create_comparison_table(self):
        """Create a pandas DataFrame comparing all experiments"""
        data = []
        
        for exp in self.all_results:
            exp_id = exp['experiment_id']
            name = exp['results']['experiment_name']
            cfg = exp['config']
            res = exp['results']
            
            row = {
                'ID': exp_id,
                'Name': name,
                'CTC': cfg.get('use_ctc', False),
                'CTC_Weight': cfg.get('ctc_weight', 0.0) if cfg.get('use_ctc', False) else 0.0,
                'Epochs': cfg.get('num_epochs', 0),
                'Beam_Size': cfg.get('beam_size', 1),
                'Learning_Rate': cfg.get('learning_rate', 0),
                'Batch_Size': cfg.get('batch_size', 0),
                'Best_Val_CER': res.get('best_val_cer', 999),
                'Final_Val_CER': res.get('final_val_cer', 999),
                'Final_Val_WER': res.get('final_val_wer', 999),
                'Training_Time_Min': res.get('training_time_seconds', 0) / 60,
                'Num_Params': res.get('num_parameters', 0)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('Best_Val_CER')
        
        return df
    
    def plot_cer_comparison(self, save_path: str = None):
        """Bar plot comparing CER across experiments"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Sort by CER
        sorted_exps = sorted(
            self.all_results,
            key=lambda x: x['results'].get('best_val_cer', 999)
        )
        
        exp_names = [exp['results']['experiment_name'] for exp in sorted_exps]
        best_cers = [exp['results'].get('best_val_cer', 999) for exp in sorted_exps]
        final_cers = [exp['results'].get('final_val_cer', 999) for exp in sorted_exps]
        
        x = np.arange(len(exp_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, best_cers, width, label='Best Val CER', alpha=0.8)
        bars2 = ax.bar(x + width/2, final_cers, width, label='Final Val CER', alpha=0.8)
        
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel('CER', fontsize=12)
        ax.set_title('Character Error Rate Comparison Across Experiments', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_training_curves(self, experiment_ids: List[int] = None, save_path: str = None):
        """Plot training curves for selected experiments"""
        if experiment_ids is None:
            # Plot all experiments
            experiments = self.all_results
        else:
            experiments = [exp for exp in self.all_results if exp['experiment_id'] in experiment_ids]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for exp in experiments:
            exp_id = exp['experiment_id']
            name = exp['results']['experiment_name']
            train_losses = exp['results'].get('train_losses', [])
            val_cers = exp['results'].get('val_cers', [])
            
            epochs = list(range(1, len(train_losses) + 1))
            
            # Plot training loss
            ax1.plot(epochs, train_losses, marker='o', label=f"#{exp_id}: {name}", linewidth=2)
            
            # Plot validation CER
            ax2.plot(epochs, val_cers, marker='s', label=f"#{exp_id}: {name}", linewidth=2)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss', fontsize=12)
        ax1.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Validation CER', fontsize=12)
        ax2.set_title('Validation CER Over Epochs', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def analyze_ctc_effect(self, save_path: str = None):
        """Analyze the effect of CTC"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate experiments with and without CTC
        with_ctc = []
        without_ctc = []
        
        for exp in self.all_results:
            cfg = exp['config']
            res = exp['results']
            cer = res.get('best_val_cer', 999)
            
            if cfg.get('use_ctc', False):
                ctc_weight = cfg.get('ctc_weight', 0.3)
                with_ctc.append((ctc_weight, cer, exp['results']['experiment_name']))
            else:
                without_ctc.append((0, cer, exp['results']['experiment_name']))
        
        # Plot
        if without_ctc:
            weights, cers, names = zip(*without_ctc)
            ax.scatter(weights, cers, s=200, alpha=0.6, label='Without CTC', marker='o')
            for w, c, n in zip(weights, cers, names):
                ax.annotate(n, (w, c), fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
        
        if with_ctc:
            weights, cers, names = zip(*with_ctc)
            ax.scatter(weights, cers, s=200, alpha=0.6, label='With CTC', marker='^')
            for w, c, n in zip(weights, cers, names):
                ax.annotate(n, (w, c), fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('CTC Weight', fontsize=12)
        ax.set_ylabel('Best Validation CER', fontsize=12)
        ax.set_title('Effect of CTC on Performance', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def analyze_beam_search_effect(self, save_path: str = None):
        """Analyze the effect of beam search"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        beam_data = {}
        
        for exp in self.all_results:
            cfg = exp['config']
            res = exp['results']
            beam_size = cfg.get('beam_size', 1)
            cer = res.get('best_val_cer', 999)
            name = res['experiment_name']
            
            if beam_size not in beam_data:
                beam_data[beam_size] = []
            beam_data[beam_size].append((cer, name))
        
        # Plot
        for beam_size, data in sorted(beam_data.items()):
            cers, names = zip(*data)
            x_positions = [beam_size + np.random.uniform(-0.1, 0.1) for _ in cers]
            ax.scatter(x_positions, cers, s=150, alpha=0.6, label=f'Beam={beam_size}')
            
            for x, c, n in zip(x_positions, cers, names):
                ax.annotate(n, (x, c), fontsize=7, alpha=0.6, xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Beam Size', fontsize=12)
        ax.set_ylabel('Best Validation CER', fontsize=12)
        ax.set_title('Effect of Beam Search on Performance', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def analyze_epochs_effect(self, save_path: str = None):
        """Analyze the effect of number of epochs"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epoch_data = {}
        
        for exp in self.all_results:
            cfg = exp['config']
            res = exp['results']
            num_epochs = cfg.get('num_epochs', 0)
            cer = res.get('best_val_cer', 999)
            name = res['experiment_name']
            
            if num_epochs not in epoch_data:
                epoch_data[num_epochs] = []
            epoch_data[num_epochs].append((cer, name))
        
        # Plot
        for num_epochs, data in sorted(epoch_data.items()):
            cers, names = zip(*data)
            x_positions = [num_epochs + np.random.uniform(-0.3, 0.3) for _ in cers]
            ax.scatter(x_positions, cers, s=150, alpha=0.6, label=f'{num_epochs} Epochs')
            
            for x, c, n in zip(x_positions, cers, names):
                ax.annotate(n, (x, c), fontsize=7, alpha=0.6, xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Number of Epochs', fontsize=12)
        ax.set_ylabel('Best Validation CER', fontsize=12)
        ax.set_title('Effect of Training Duration on Performance', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        report_dir = self.results_dir / "analysis"
        report_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("GENERATING EXPERIMENT ANALYSIS REPORT")
        print("="*80 + "\n")
        
        # 1. Comparison table
        print("📊 Creating comparison table...")
        df = self.create_comparison_table()
        
        # Save to CSV
        csv_path = report_dir / "comparison_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"   Saved to {csv_path}")
        
        # Print top 5
        print("\n🏆 Top 5 Experiments by CER:")
        print(df.head(5).to_string(index=False))
        
        # 2. CER comparison plot
        print("\n📈 Creating CER comparison plot...")
        self.plot_cer_comparison(save_path=str(report_dir / "cer_comparison.png"))
        
        # 3. Training curves (top 5 experiments)
        print("\n📉 Creating training curves for top 5 experiments...")
        top_5_ids = df.head(5)['ID'].tolist()
        self.plot_training_curves(
            experiment_ids=top_5_ids,
            save_path=str(report_dir / "top5_training_curves.png")
        )
        
        # 4. CTC effect analysis
        print("\n🔬 Analyzing CTC effect...")
        self.analyze_ctc_effect(save_path=str(report_dir / "ctc_effect.png"))
        
        # 5. Beam search effect analysis
        print("\n🔍 Analyzing beam search effect...")
        self.analyze_beam_search_effect(save_path=str(report_dir / "beam_search_effect.png"))
        
        # 6. Epochs effect analysis
        print("\n⏱️  Analyzing training duration effect...")
        self.analyze_epochs_effect(save_path=str(report_dir / "epochs_effect.png"))
        
        # 7. Summary statistics
        print("\n📝 Generating summary statistics...")
        summary = self._generate_summary_stats(df)
        summary_path = report_dir / "summary_statistics.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"   Saved to {summary_path}")
        print(summary)
        
        print("\n" + "="*80)
        print(f"✅ REPORT COMPLETE! All files saved to: {report_dir}")
        print("="*80)
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> str:
        """Generate summary statistics"""
        summary = "\n" + "="*80 + "\n"
        summary += "SUMMARY STATISTICS\n"
        summary += "="*80 + "\n\n"
        
        # Overall statistics
        summary += "Overall Statistics:\n"
        summary += f"  Total experiments: {len(df)}\n"
        summary += f"  Best CER: {df['Best_Val_CER'].min():.4f}\n"
        summary += f"  Worst CER: {df['Best_Val_CER'].max():.4f}\n"
        summary += f"  Average CER: {df['Best_Val_CER'].mean():.4f}\n"
        summary += f"  Std Dev CER: {df['Best_Val_CER'].std():.4f}\n\n"
        
        # CTC vs No CTC
        with_ctc = df[df['CTC'] == True]
        without_ctc = df[df['CTC'] == False]
        
        if len(with_ctc) > 0 and len(without_ctc) > 0:
            summary += "CTC Effect:\n"
            summary += f"  With CTC - Avg CER: {with_ctc['Best_Val_CER'].mean():.4f} (n={len(with_ctc)})\n"
            summary += f"  Without CTC - Avg CER: {without_ctc['Best_Val_CER'].mean():.4f} (n={len(without_ctc)})\n"
            improvement = ((without_ctc['Best_Val_CER'].mean() - with_ctc['Best_Val_CER'].mean()) / 
                          without_ctc['Best_Val_CER'].mean() * 100)
            summary += f"  Improvement: {improvement:.2f}%\n\n"
        
        # Beam search effect
        beam_1 = df[df['Beam_Size'] == 1]
        beam_gt_1 = df[df['Beam_Size'] > 1]
        
        if len(beam_1) > 0 and len(beam_gt_1) > 0:
            summary += "Beam Search Effect:\n"
            summary += f"  Greedy (beam=1) - Avg CER: {beam_1['Best_Val_CER'].mean():.4f} (n={len(beam_1)})\n"
            summary += f"  Beam search (beam>1) - Avg CER: {beam_gt_1['Best_Val_CER'].mean():.4f} (n={len(beam_gt_1)})\n"
            improvement = ((beam_1['Best_Val_CER'].mean() - beam_gt_1['Best_Val_CER'].mean()) / 
                          beam_1['Best_Val_CER'].mean() * 100)
            summary += f"  Improvement: {improvement:.2f}%\n\n"
        
        # Best experiment details
        best_exp = df.iloc[0]
        summary += "🏆 Best Experiment Details:\n"
        summary += f"  ID: {best_exp['ID']}\n"
        summary += f"  Name: {best_exp['Name']}\n"
        summary += f"  CER: {best_exp['Best_Val_CER']:.4f}\n"
        summary += f"  WER: {best_exp['Final_Val_WER']:.4f}\n"
        summary += f"  Configuration:\n"
        summary += f"    - CTC: {best_exp['CTC']}\n"
        summary += f"    - CTC Weight: {best_exp['CTC_Weight']}\n"
        summary += f"    - Epochs: {best_exp['Epochs']}\n"
        summary += f"    - Beam Size: {best_exp['Beam_Size']}\n"
        summary += f"    - Learning Rate: {best_exp['Learning_Rate']}\n"
        summary += f"    - Batch Size: {best_exp['Batch_Size']}\n"
        summary += f"  Training Time: {best_exp['Training_Time_Min']:.1f} minutes\n"
        
        return summary


def main():
    """Main visualization function"""
    print("📊 Experiment Results Analyzer")
    print("="*80)
    
    try:
        visualizer = ExperimentVisualizer(results_dir="experiment_results")
        visualizer.generate_report()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please run experiments first using experiment_runner.py")
        return
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
