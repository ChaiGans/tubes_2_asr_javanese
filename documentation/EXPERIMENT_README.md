# Experiment Framework - Quick Reference

## 🎯 What You Asked For

> "I want to make experiments for all possibilities in the current model, so I can see the score improvement per changes. For example:
> - If I add CTC, here is the effect
> - If I tune the epoch, here is the effect  
> - If I use beam, I got this score"

**✅ Solution Created!** This framework systematically tests all configurations and shows you exactly what each change does to your scores.

---

## 📦 Files Created

### 1. **`experiment_runner.py`** (Main Experiment Runner)
**Purpose**: Automatically run 12+ predefined experiments testing different configurations

**What it tests**:
- CTC: On/Off, different weights (0.3, 0.5)
- Epochs: 5, 10, 20
- Beam search: Greedy (1), Beam (3, 5)
- Learning rates: 5e-4, 1e-3, 3e-3
- Batch sizes: 8, 16

**Usage**:
```bash
python experiment_runner.py
```

**Output**: All results saved to `experiment_results/`

---

### 2. **`visualize_experiments.py`** (Results Analyzer)
**Purpose**: Create charts and tables showing what each change does

**What it generates**:
- 📊 Comparison table showing all experiments
- 📈 CER/WER comparison bar charts
- 📉 Training curves over epochs
- 🔬 Effect analysis: "Adding CTC improved CER by X%"
- 📝 Summary statistics

**Usage**:
```bash
python visualize_experiments.py
```

**Output**: Charts and analysis saved to `experiment_results/analysis/`

---

### 3. **`quick_experiment.py`** (Interactive Launcher)
**Purpose**: Run custom experiments interactively or test specific parameters

**Features**:
- Interactive menu
- Compare two configurations head-to-head
- Test different CTC weights
- Test different beam sizes
- Run custom single experiments

**Usage**:
```bash
python quick_experiment.py
```

---

### 4. **`EXPERIMENT_GUIDE.md`** (Complete Guide)
**Purpose**: Detailed documentation and examples

**Contains**:
- Step-by-step instructions
- How to interpret results
- Customization guide
- Tips and best practices
- Example workflows

---

## 🚀 How to Use

### Method 1: Run All Experiments (Recommended First)

```bash
# Step 1: Run all predefined experiments
python experiment_runner.py

# Step 2: Analyze results
python visualize_experiments.py

# Step 3: Check the summary
cat experiment_results/summary.txt
```

This will show you tables like:

```
ID  CTC   Epochs  Beam  LR      Val CER   Effect
1   Yes   10      5     1e-3    0.2145    CTC + Beam: -35% vs baseline!
2   Yes   10      3     1e-3    0.2198    CTC + Beam: -34% vs baseline!
3   No    10      5     1e-3    0.2543    Beam only: -24% vs baseline!
4   No    10      1     1e-3    0.3324    Baseline
...
```

### Method 2: Quick Custom Experiment

```bash
# Interactive mode
python quick_experiment.py
```

Or edit the script to run specific tests:

```python
# Test CTC effect
python quick_experiment.py
# Choose option 3: Test different CTC weights
```

### Method 3: Run One Specific Experiment

```python
from quick_experiment import run_quick_experiment

# Test: "What happens if I add CTC?"
run_quick_experiment(
    name="Test CTC",
    use_ctc=True,
    ctc_weight=0.3,
    num_epochs=10
)
```

---

## 📊 Example Output

After running experiments, you'll get clear answers:

### Q: "If I add CTC, here is the effect"
**Answer from results**:
```
Without CTC - Avg CER: 0.3324
With CTC    - Avg CER: 0.2543
Improvement: 23.5% ✅
```

### Q: "If I tune the epochs, here is the effect"
**Answer from results**:
```
5 Epochs  - CER: 0.3012
10 Epochs - CER: 0.2543  ✅ Better!
20 Epochs - CER: 0.2198  ✅✅ Best!
```

### Q: "If I use beam, I got this score"
**Answer from results**:
```
Greedy (beam=1) - CER: 0.2876
Beam=3          - CER: 0.2543  ✅ Better!
Beam=5          - CER: 0.2198  ✅✅ Best!
```

---

## 📁 Results Structure

```
experiment_results/
├── all_experiments.json          # All experiment data
├── experiment_001.json            # Individual results
├── experiment_002.json
├── ...
├── summary.txt                    # Quick text summary
└── analysis/
    ├── comparison_table.csv       # Excel-ready comparison
    ├── cer_comparison.png         # Bar chart: all experiments
    ├── top5_training_curves.png   # Line chart: best 5 experiments
    ├── ctc_effect.png             # Scatter plot: CTC analysis
    ├── beam_search_effect.png     # Scatter plot: Beam analysis
    ├── epochs_effect.png          # Scatter plot: Epochs analysis
    └── summary_statistics.txt     # Detailed statistics
```

---

## 🎯 Your Current Baseline

From `my_results.json`:
- **Current CER**: 0.3324
- **Current WER**: 0.4206

**Goal**: Find configurations that beat this!

---

## ⚡ Quick Commands

```bash
# Run all experiments (takes time, but comprehensive)
python experiment_runner.py

# Analyze results and create visualizations
python visualize_experiments.py

# Run a quick test
python quick_experiment.py

# View summary
cat experiment_results/summary.txt

# View best experiment details
cat experiment_results/experiment_001.json
```

---

## 💡 Recommended Workflow

### Day 1: Quick Exploration (1-2 hours)
```bash
# Test basic ideas with 5 epochs (fast)
python quick_experiment.py
# Choose option 3: Test CTC weights
# Choose option 4: Test beam sizes
```

### Day 2: Systematic Testing (4-6 hours)
```bash
# Run full experiment suite
python experiment_runner.py
```

### Day 3: Analysis & Refinement
```bash
# Analyze results
python visualize_experiments.py

# Review charts in experiment_results/analysis/

# Run best configuration with more epochs
python quick_experiment.py
# Choose option 1 and use best config from results
```

---

## 🔧 Customization

### Add Your Own Experiment

Edit `experiment_runner.py`, find `define_experiments()`, add:

```python
experiments.append({
    "name": "My Custom Test",
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

---

## ✅ What This Gives You

1. **Systematic Testing**: All major configurations tested automatically
2. **Clear Comparisons**: Side-by-side charts showing exactly what each change does
3. **Data-Driven Decisions**: Numbers proving which changes help
4. **Reproducibility**: All configs saved, can re-run any experiment
5. **Publication-Ready Charts**: Professional visualizations for reports

---

## 📞 Help

- Read `EXPERIMENT_GUIDE.md` for detailed instructions
- Check `experiment_results/summary.txt` for quick results
- Look at `experiment_results/analysis/` for visualizations

---

## 🎉 Summary

**You now have**:
- ✅ Automated experiment runner testing all configurations
- ✅ Visualization tools showing effect of each change
- ✅ Interactive tools for custom experiments
- ✅ Complete documentation

**Next steps**:
1. Run `python experiment_runner.py` to start testing
2. Run `python visualize_experiments.py` to see results
3. Find your best configuration!
4. Use it to train your final model

Good luck! 🚀
