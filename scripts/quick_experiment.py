"""
Quick experiment launcher - customize and run specific experiments
"""

from scripts.experiment_runner import ExperimentRunner, ExperimentTracker
from config import Config


def run_quick_experiment(
    name: str,
    use_ctc: bool = False,
    ctc_weight: float = 0.3,
    num_epochs: int = 5,
    beam_size: int = 1,
    learning_rate: float = 1e-3,
    batch_size: int = 8
):
    """
    Run a single experiment with custom parameters
    
    Example:
        run_quick_experiment(
            name="Test CTC",
            use_ctc=True,
            num_epochs=10,
            beam_size=5
        )
    """
    print(f"\n🚀 Running Quick Experiment: {name}")
    
    base_config = Config()
    tracker = ExperimentTracker()
    runner = ExperimentRunner(base_config, tracker)
    
    config_overrides = {
        "use_ctc": use_ctc,
        "ctc_weight": ctc_weight,
        "num_epochs": num_epochs,
        "beam_size": beam_size,
        "learning_rate": learning_rate,
        "batch_size": batch_size
    }
    
    results = runner.run_single_experiment(
        experiment_name=name,
        config_overrides=config_overrides
    )
    
    print(f"\n✅ Experiment '{name}' completed!")
    print(f"   Best CER: {results['best_val_cer']:.4f}")
    print(f"   Final CER: {results['final_val_cer']:.4f}")
    
    return results


def compare_two_configs():
    """
    Example: Compare two specific configurations
    """
    print("\n" + "="*80)
    print("COMPARING TWO CONFIGURATIONS")
    print("="*80)
    
    # Config A: Baseline
    print("\n📊 Running Configuration A (Baseline)...")
    results_a = run_quick_experiment(
        name="Config A: Baseline",
        use_ctc=False,
        num_epochs=5,
        beam_size=1,
        learning_rate=1e-3
    )
    
    # Config B: With improvements
    print("\n📊 Running Configuration B (With CTC + Beam)...")
    results_b = run_quick_experiment(
        name="Config B: CTC + Beam",
        use_ctc=True,
        ctc_weight=0.3,
        num_epochs=5,
        beam_size=5,
        learning_rate=1e-3
    )
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"\nConfig A (Baseline):")
    print(f"  Best CER: {results_a['best_val_cer']:.4f}")
    print(f"  Final CER: {results_a['final_val_cer']:.4f}")
    
    print(f"\nConfig B (CTC + Beam):")
    print(f"  Best CER: {results_b['best_val_cer']:.4f}")
    print(f"  Final CER: {results_b['final_val_cer']:.4f}")
    
    improvement = (results_a['best_val_cer'] - results_b['best_val_cer']) / results_a['best_val_cer'] * 100
    print(f"\n{'🎉' if improvement > 0 else '⚠️'} Improvement: {improvement:+.2f}%")


def test_ctc_weights():
    """
    Example: Test different CTC weights
    """
    print("\n" + "="*80)
    print("TESTING DIFFERENT CTC WEIGHTS")
    print("="*80)
    
    ctc_weights = [0.1, 0.3, 0.5, 0.7]
    results = []
    
    for weight in ctc_weights:
        print(f"\n📊 Testing CTC weight = {weight}...")
        result = run_quick_experiment(
            name=f"CTC weight {weight}",
            use_ctc=True,
            ctc_weight=weight,
            num_epochs=5,
            beam_size=1
        )
        results.append((weight, result['best_val_cer']))
    
    # Print summary
    print("\n" + "="*80)
    print("CTC WEIGHT COMPARISON")
    print("="*80)
    print(f"\n{'CTC Weight':<15} {'Best CER':<10}")
    print("-"*25)
    
    for weight, cer in sorted(results, key=lambda x: x[1]):
        print(f"{weight:<15.1f} {cer:<10.4f}")
    
    best_weight, best_cer = min(results, key=lambda x: x[1])
    print(f"\n🏆 Best CTC weight: {best_weight} (CER: {best_cer:.4f})")


def test_beam_sizes():
    """
    Example: Test different beam sizes
    """
    print("\n" + "="*80)
    print("TESTING DIFFERENT BEAM SIZES")
    print("="*80)
    
    beam_sizes = [1, 3, 5, 7, 10]
    results = []
    
    for beam in beam_sizes:
        print(f"\n📊 Testing beam size = {beam}...")
        result = run_quick_experiment(
            name=f"Beam size {beam}",
            use_ctc=True,
            ctc_weight=0.3,
            num_epochs=5,
            beam_size=beam
        )
        results.append((beam, result['best_val_cer']))
    
    # Print summary
    print("\n" + "="*80)
    print("BEAM SIZE COMPARISON")
    print("="*80)
    print(f"\n{'Beam Size':<15} {'Best CER':<10}")
    print("-"*25)
    
    for beam, cer in sorted(results, key=lambda x: x[1]):
        print(f"{beam:<15d} {cer:<10.4f}")
    
    best_beam, best_cer = min(results, key=lambda x: x[1])
    print(f"\n🏆 Best beam size: {best_beam} (CER: {best_cer:.4f})")


def main():
    """Interactive menu"""
    print("\n" + "="*80)
    print("QUICK EXPERIMENT LAUNCHER")
    print("="*80)
    
    print("\nChoose an option:")
    print("  1. Run a custom single experiment")
    print("  2. Compare two configurations")
    print("  3. Test different CTC weights")
    print("  4. Test different beam sizes")
    print("  5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        # Custom experiment
        print("\n" + "-"*80)
        name = input("Experiment name: ").strip()
        use_ctc = input("Use CTC? (y/n): ").strip().lower() == 'y'
        
        if use_ctc:
            ctc_weight = float(input("CTC weight (0.0-1.0, default 0.3): ").strip() or "0.3")
        else:
            ctc_weight = 0.3
        
        num_epochs = int(input("Number of epochs (default 5): ").strip() or "5")
        beam_size = int(input("Beam size (1=greedy, default 1): ").strip() or "1")
        learning_rate = float(input("Learning rate (default 1e-3): ").strip() or "1e-3")
        batch_size = int(input("Batch size (default 8): ").strip() or "8")
        
        run_quick_experiment(
            name=name,
            use_ctc=use_ctc,
            ctc_weight=ctc_weight,
            num_epochs=num_epochs,
            beam_size=beam_size,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        
    elif choice == "2":
        compare_two_configs()
        
    elif choice == "3":
        test_ctc_weights()
        
    elif choice == "4":
        test_beam_sizes()
        
    elif choice == "5":
        print("\n👋 Goodbye!")
        return
    
    else:
        print("\n❌ Invalid choice")


if __name__ == "__main__":
    # Example usage - uncomment what you want to run:
    
    # Option 1: Interactive menu
    main()
    
    # Option 2: Quick single experiment
    # run_quick_experiment(
    #     name="My Test",
    #     use_ctc=True,
    #     num_epochs=10,
    #     beam_size=5
    # )
    
    # Option 3: Compare configurations
    # compare_two_configs()
    
    # Option 4: Test CTC weights
    # test_ctc_weights()
    
    # Option 5: Test beam sizes
    # test_beam_sizes()
