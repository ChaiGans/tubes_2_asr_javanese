# Experiment Framework for Javanese ASR

This framework allows you to systematically test different model configurations and track performance improvements.

## 📁 Files

- **`experiment_runner.py`**: Main script to run systematic experiments
- **`visualize_experiments.py`**: Script to analyze and visualize results
- **`EXPERIMENT_GUIDE.md`**: This guide

## 🚀 Quick Start

### 1. Run Experiments

The `experiment_runner.py` script will automatically run multiple experiments with different configurations:

```bash
python experiment_runner.py
```

This will test:
- ✅ **CTC effect**: With/without CTC, different CTC weights
- ✅ **Epoch tuning**: 5, 10, 20 epochs
- ✅ **Beam search**: Greedy vs beam search (beam=3, 5)
- ✅ **Learning rate**: 5e-4, 1e-3, 3e-3
- ✅ **Batch size**: 8, 16

Each experiment will:
1. Train a model with specific configuration
2. Evaluate on validation set
3. Save results to `experiment_results/`

### 2. Analyze Results

After running experiments, visualize and compare them:

```bash
python visualize_experiments.py
```

This generates:
- 📊 Comparison tables (CSV)
- 📈 CER/WER comparison charts
- 📉 Training curves
- 🔬 Effect analysis (CTC, beam search, epochs)
- 📝 Summary statistics

All outputs saved to `experiment_results/analysis/`

## 📊 Understanding Results

### Result Files

```
experiment_results/
├── all_experiments.json          # All experiment data
├── experiment_001.json            # Individual experiment 1
├── experiment_002.json            # Individual experiment 2
├── ...
├── summary.txt                    # Quick summary
└── analysis/                      # Visualizations
    ├── comparison_table.csv
    ├── cer_comparison.png
    ├── top5_training_curves.png
    ├── ctc_effect.png
    ├── beam_search_effect.png
    ├── epochs_effect.png
    └── summary_statistics.txt
```

### Key Metrics

- **CER (Character Error Rate)**: Lower is better. Measures character-level errors.
- **WER (Word Error Rate)**: Lower is better. Measures word-level errors.
- **Best Val CER**: Best validation CER during training
- **Final Val CER**: Validation CER at the final epoch

## 🔧 Customizing Experiments

### Add Your Own Experiments

Edit `define_experiments()` in `experiment_runner.py`:

```python
experiments.append({
    "name": "My Custom Experiment",
    "config": {
        "use_ctc": True,
        "ctc_weight": 0.4,
        "num_epochs": 15,
        "beam_size": 7,
        "learning_rate": 2e-3,
        "batch_size": 12
    }
})
```

### Available Configuration Options

From `config.py`, you can modify:

**Training:**
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `grad_clip_norm`: Gradient clipping norm
- `teacher_forcing_ratio`: Teacher forcing ratio

**Model Architecture:**
- `encoder_hidden_size`: Encoder hidden dimension
- `encoder_num_layers`: Number of encoder layers
- `decoder_dim`: Decoder dimension
- `attention_dim`: Attention dimension
- `embedding_dim`: Embedding dimension
- `dropout`: Dropout rate

**CTC Settings:**
- `use_ctc`: Enable/disable CTC
- `ctc_weight`: Weight for CTC loss (λ)

**Decoding:**
- `beam_size`: Beam size (1 = greedy)
- `max_decode_len`: Maximum decode length

**Features:**
- `n_mels`: Number of mel filterbanks
- `apply_spec_augment`: Enable SpecAugment
- `speed_perturb`: Enable speed perturbation

## 📈 Interpreting Results

### Example Output

```
ID    CTC    Epochs   Beam   LR         Val CER    Val WER   
1     Yes    10       5      1e-3       0.2145     0.2574    
2     Yes    10       3      1e-3       0.2198     0.2638    
3     No     10       5      1e-3       0.2543     0.3052    
4     Yes    5        1      1e-3       0.2876     0.3451    
...
```

**Analysis:**
- Experiment #1 achieves best CER (0.2145)
- CTC + Beam Search (5) gives best results
- CTC reduces CER by ~15% vs no CTC
- Beam search improves over greedy decoding

### Effect Analysis

The visualizer will show:

1. **CTC Effect**: Compare CER with/without CTC
   - If CTC experiments cluster lower → CTC helps
   
2. **Beam Search Effect**: Compare by beam size
   - Larger beam → better CER (usually)
   - Diminishing returns after beam=5-7
   
3. **Epoch Effect**: Training duration impact
   - More epochs → lower CER (until overfitting)
   - Look for plateau to find optimal epochs

## 💡 Tips

### 1. Start Small
- Run quick experiments (5 epochs) to test ideas
- Scale up promising configurations

### 2. Sequential Testing
- Test one variable at a time
- Example: First find best CTC weight, then tune beam size

### 3. Resource Management
- Reduce `batch_size` if running out of memory
- Use `beam_size=1` (greedy) for faster training
- Set `num_epochs` lower for initial exploration

### 4. Comparing with Baseline
Your current `my_results.json` shows:
- avg_cer: 0.3324
- avg_wer: 0.4206

Compare experiment results to see improvements!

## 🎯 Example Workflow

1. **Initial Exploration** (Fast iterations)
```python
# Test basic configurations with 5 epochs
- Baseline (no CTC, greedy)
- With CTC (ctc_weight=0.3)
- Different beam sizes
```

2. **Deep Dive** (Promising configs)
```python
# Take best 2-3 configs, train longer
- Best config + 20 epochs
- Best config + different LR
```

3. **Final Tuning**
```python
# Fine-tune the best configuration
- Adjust CTC weight
- Try different beam sizes
- Tune learning rate schedule
```

## 📞 Quick Commands

```bash
# Run all experiments
python experiment_runner.py

# Analyze results
python visualize_experiments.py

# View summary quickly
cat experiment_results/summary.txt

# Check specific experiment
cat experiment_results/experiment_001.json
```

## 🔍 Advanced: Manual Experiment

To run a single experiment manually:

```python
from experiment_runner import ExperimentRunner, ExperimentTracker
from config import Config

base_config = Config()
tracker = ExperimentTracker()
runner = ExperimentRunner(base_config, tracker)

results = runner.run_single_experiment(
    experiment_name="My Test",
    config_overrides={
        "use_ctc": True,
        "num_epochs": 15,
        "beam_size": 5
    }
)
```

## 📚 Understanding Your Question

You asked: "if i add ctc, here is the effect. if i tune the epoch here is the effect. if i use beam i got this score."

This framework does exactly that! Each experiment isolates a variable:

- **CTC Effect**: Compare experiments with `use_ctc=True` vs `False`
- **Epoch Effect**: Compare experiments with different `num_epochs`
- **Beam Effect**: Compare experiments with different `beam_size`

The visualizer will show you graphs and tables making these effects clear!

## 🎉 Expected Results

After running the experiments, you'll get charts like:

```
CER Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline (No CTC): ████████░░ 0.3324
With CTC:          ██████░░░░ 0.2543  ✓ Better!
CTC + Beam(5):     ████░░░░░░ 0.2145  ✓✓ Best!
```

This makes it easy to see which changes improve performance!
