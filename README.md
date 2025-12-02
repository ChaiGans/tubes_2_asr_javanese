# LAS-Style Seq2Seq ASR for Javanese

A from-scratch implementation of a Listen, Attend and Spell (LAS) style sequence-to-sequence Automatic Speech Recognition (ASR) model for low-resource Javanese language (~3-4 hours of audio).

## Features

- **Pyramidal BiLSTM Encoder**: 2-3 BiLSTM layers with pyramidal time reduction
- **Location-Sensitive Attention**: Chorowski-style attention mechanism
- **LSTM Decoder with Input Feeding**: Luong-style input feeding
- **Optional CTC Head**: Joint CTC+Attention training
- **SpecAugment & CMVN**: Data augmentation and normalization
- **Character-Based**: Uses character-level vocabulary for Javanese

## Model Architecture

```
Input Audio (16kHz) 
  ↓
Log-Mel Features (80-dim)
  ↓
Pyramidal BiLSTM Encoder (3 layers, hidden=128)
  ↓
Location-Sensitive Attention
  ↓
LSTM Decoder (1 layer, hidden=256)
  ↓
Character Predictions
```

## Installation

### Requirements

```bash
pip install torch torchaudio editdistance tqdm
```

### Dependencies

- Python 3.7+
- PyTorch 1.10+
- torchaudio
- editdistance (for CER computation)
- tqdm (for progress bars)

## Data Format

### Audio Files
- Format: WAV, PCM 16-bit, mono, 16 kHz sample rate
- Location: `audio_input/` directory
- Naming: `Speaker01_m_nn_utt01.wav` (capital S), `speaker01_m_nn_utt01.wav`, or with dashes

### Transcripts
- Format: CSV with header `SentenceID,Transcript,Device`
- Location: `transcripts.csv`
- Example:
  ```
  SentenceID,Transcript,Device
  speaker01_m_nn_utt01,aku libur sedina merga ana banjir,
  speaker01_m_nn_utt02,aku pengin mangan dhawet nang omah,
  ```

## Usage

### 1. Build Vocabulary

The vocabulary is automatically built from the transcript file during the first training run:

```python
from vocab import build_vocab_from_file

vocab = build_vocab_from_file("transcripts.csv", save_path="vocab.json")
```

### 2. Training

#### Basic Training
```bash
python train.py
```

#### Custom Configuration
Edit `config.py` to adjust hyperparameters:
- `batch_size`: Batch size (default: 8)
- `num_epochs`: Number of training epochs (default: 100)
- `learning_rate`: Learning rate (default: 1e-3)
- `use_ctc`: Enable joint CTC+attention (default: False)
- `apply_spec_augment`: Enable SpecAugment (default: True)

### 3. Inference

#### Single Audio File
```bash
python inference.py --audio path/to/audio.wav --checkpoint checkpoints/best_model.pt
```

#### Directory of Audio Files
```bash
python inference.py --audio path/to/audio_dir/ --checkpoint checkpoints/best_model.pt
```

#### With Beam Search
```bash
python inference.py --audio audio.wav --decoder beam --beam_size 10
```

## File Structure

```
.
├── model.py                 # Model architecture (Encoder, Attention, Decoder, Seq2SeqASR)
├── features.py              # Feature extraction (Log-Mel, CMVN, SpecAugment)
├── vocab.py                 # Vocabulary builder
├── dataset.py               # Dataset and DataLoader
├── metrics.py               # CER and WER metrics
├── decoder.py               # Greedy and Beam Search decoders
├── train.py                 # Training script
├── inference.py             # Inference script
├── config.py                # Configuration
├── utils.py                 # Utility functions
├── audio_input/             # Audio files directory
├── transcripts.csv          # Transcript file
├── vocab.json               # Generated vocabulary
└── checkpoints/             # Model checkpoints
```

## Model Details

### Encoder
- **Type**: Pyramidal BiLSTM
- **Layers**: 3 BiLSTM layers
- **Hidden Size**: 128 per direction (256 total)
- **Time Reduction**: 2 pyramidal reductions (4x total)
- **Dropout**: 0.3

### Attention
- **Type**: Location-Sensitive Additive Attention
- **Mechanism**: Combines decoder state, encoder outputs, and convolved previous attention weights
- **Attention Dim**: 128
- **Location Filters**: 10 filters, kernel size 5

### Decoder
- **Type**: LSTM with Input Feeding
- **Layers**: 1 LSTM layer
- **Hidden Size**: 256
- **Embedding Dim**: 64
- **Input**: [embedding(y_{t-1}); context_{t-1}]
- **Output**: Linear projection to vocabulary size

### Training
- **Optimizer**: Adam (lr=1e-3)
- **Gradient Clipping**: Norm 5.0
- **Teacher Forcing**: 1.0 (always use ground truth)
- **Batch Size**: 8
- **Validation Split**: 10%

## Example Training Output

```
Using device: cuda
Loading vocabulary from vocab.json
Vocabulary size: 87
Loading dataset...
Loaded 2100 utterances from transcripts.csv
Train size: 1890, Val size: 210
Creating model...
Model parameters: 1,234,567

=== Starting Training for 100 epochs ===

Epoch 1/100 [Train]: 100%|████| 236/236 [01:23<00:00,  2.84it/s, loss=4.1234]
Validation: 100%|████████████| 27/27 [00:05<00:00,  5.12it/s, loss=3.8765, cer=0.8234]

Epoch 1/100 (88.5s):
  Train Loss: 4.1234
  Val Loss:   3.8765
  Val CER:    0.8234
  *** New best model (CER: 0.8234) ***
```

## Testing the Model

### Quick Overfitting Test
To verify the model can learn, train on a small subset:

```python
# In train.py, modify the dataset split
train_dataset = torch.utils.data.Subset(full_dataset, range(5))
val_dataset = torch.utils.data.Subset(full_dataset, range(5))
```

The model should achieve near-zero loss and CER on these 5 samples.

## Customization

### Enable Joint CTC+Attention
In `config.py`:
```python
use_ctc = True
ctc_weight = 0.3  # Weight for CTC loss
```

### Adjust Model Size
For larger datasets:
```python
encoder_hidden_size = 256  # Increase to 256
encoder_num_layers = 4     # Add more layers
decoder_dim = 512          # Larger decoder
```

### Enable Speed Perturbation
In `config.py`:
```python
speed_perturb = True  # Augment with 0.9x and 1.1x speed
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in `config.py`
- Reduce `encoder_hidden_size` or `decoder_dim`
- Use CPU instead of GPU

### Poor Performance
- Increase `num_epochs` (e.g., 200-300 for small datasets)
- Enable `use_ctc = True` for joint training
- Ensure `apply_spec_augment = True`
- Check data quality and vocabulary coverage

### Slow Training
- Increase `batch_size` if GPU memory allows
- Set `num_workers > 0` in DataLoader (Linux/Mac only)
- Use GPU with CUDA

## Citation

If you use this code, please cite the following papers:

```bibtex
@article{chan2016listen,
  title={Listen, attend and spell: A neural network for large vocabulary conversational speech recognition},
  author={Chan, William and Jaitly, Navdeep and Le, Quoc and Vinyals, Oriol},
  journal={ICASSP},
  year={2016}
}

@article{chorowski2015attention,
  title={Attention-based models for speech recognition},
  author={Chorowski, Jan K and Bahdanau, Dzmitry and Serdyuk, Dmitriy and Cho, Kyunghyun and Bengio, Yoshua},
  journal={NeurIPS},
  year={2015}
}
```

## License

This implementation is for educational purposes.
