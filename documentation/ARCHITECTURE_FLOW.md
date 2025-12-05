# Javanese ASR - Complete Architecture Flow with Dimensions

## Overall Model Architecture with Dimension Transformations

```mermaid
graph TD
    Start([Audio File .wav]) --> Load[Load Audio]
    Load --> Wave[Waveform<br/><b>[time]</b><br/>16kHz]
    
    Wave --> FE[LogMel Feature Extractor<br/>win=25ms, hop=10ms]
    FE --> LogMel[Log-Mel Features<br/><b>[T, 80]</b><br/>T ≈ time*100 frames/sec]
    
    LogMel --> CMVN[CMVN Normalization]
    CMVN --> Norm[Normalized Features<br/><b>[T, 80]</b>]
    
    Norm --> SpecAug{Training?}
    SpecAug -->|Yes| Aug[SpecAugment<br/>Freq & Time Masking]
    SpecAug -->|No| Skip[Skip Augmentation]
    Aug --> Input[Input Features<br/><b>[B, T, 80]</b>]
    Skip --> Input
    
    Input --> Enc[<b>ENCODER: Pyramidal BiLSTM</b>]
    
    subgraph Encoder [" "]
        direction TB
        E1[Bottom BiLSTM<br/>input_dim=80, hidden=128]
        E1 --> E1Out[<b>[B, T, 256]</b><br/>256 = 128*2 bidirectional]
        
        E1Out --> Pyr1[Pyramid Reduction 1<br/>Concatenate adjacent frames]
        Pyr1 --> P1Out[<b>[B, T/2, 512]</b><br/>512 = 256*2 concat]
        
        P1Out --> E2[Pyramid BiLSTM 1<br/>input_dim=512, hidden=128]
        E2 --> E2Out[<b>[B, T/2, 256]</b>]
        
        E2Out --> Pyr2[Pyramid Reduction 2<br/>Concatenate adjacent frames]
        Pyr2 --> P2Out[<b>[B, T/4, 512]</b>]
        
        P2Out --> E3[Pyramid BiLSTM 2<br/>input_dim=512, hidden=128]
        E3 --> E3Out[<b>[B, T/4, 256]</b><br/>Final encoder output]
    end
    
    Enc --> EncOut[Encoder Outputs<br/><b>[B, T', 256]</b><br/>where T' = T/4]
    
    EncOut --> CTC{Use CTC?}
    CTC -->|Yes| CTCHead[CTC Head: Linear<br/>in=256, out=vocab_size]
    CTCHead --> CTCOut[CTC Logits<br/><b>[B, T', V]</b><br/>V = vocab_size]
    CTCOut --> CTCLoss[CTC Loss]
    CTC -->|Always| Dec
    
    EncOut --> Dec[<b>DECODER: Attention-based LSTM</b>]
    
    subgraph Decoder [" "]
        direction TB
        D0[Previous Token<br/><b>[B]</b> indices]
        D0 --> Emb[Embedding Layer<br/>vocab_size → 64]
        Emb --> EmbOut[<b>[B, 64]</b>]
        
        PrevCtx[Previous Context<br/><b>[B, 256]</b>]
        
        EmbOut --> Concat[Concatenate]
        PrevCtx --> Concat
        Concat --> ConcatOut[<b>[B, 320]</b><br/>320 = 64 + 256]
        
        ConcatOut --> LSTM[LSTM Cell<br/>input=320, hidden=256]
        LSTM --> LSTMOut[Hidden State<br/><b>[B, 256]</b>]
        
        LSTMOut --> Attn[Location-Sensitive Attention]
        EncOutDec[Encoder Outputs<br/><b>[B, T', 256]</b>] --> Attn
        PrevAttn[Previous Attention<br/><b>[B, T']</b>] --> Attn
        
        subgraph AttentionMech [" "]
            direction TB
            A1[Location Features<br/>Conv1d on prev attention]
            A1 --> A1Out[<b>[B, T', 10]</b>]
            
            A2[Query: decoder state<br/><b>[B, 256]</b>]
            A3[Key: encoder outputs<br/><b>[B, T', 256]</b>]
            A1Out --> A4[Project to attention_dim]
            A2 --> A5[Project to attention_dim]
            A3 --> A6[Project to attention_dim]
            
            A4 --> Energy[Compute Energy<br/><b>[B, T', 128]</b>]
            A5 --> Energy
            A6 --> Energy
            
            Energy --> Score[Attention Scores<br/><b>[B, T']</b>]
            Score --> Context[Attention Context<br/><b>[B, 256]</b>]
        end
        
        Attn --> AttnOut[Context Vector<br/><b>[B, 256]</b>]
        AttnOut --> Proj[Projection: Linear<br/>256 → vocab_size]
        Proj --> LogitsOut[Step Logits<br/><b>[B, V]</b>]
        
        LogitsOut --> Loop{More steps?}
        Loop -->|Yes| D0
        Loop -->|No| FinalLogits
    end
    
    Dec --> FinalLogits[All Logits<br/><b>[B, L, V]</b><br/>L = target length]
    
    FinalLogits --> CELoss[Cross-Entropy Loss]
    CTCLoss --> Combined{Joint CTC-Attention?}
    CELoss --> Combined
    
    Combined -->|Yes| Final[Total Loss<br/>λ*CTC + (1-λ)*CE<br/>λ = ctc_weight]
    Combined -->|No| Final2[CE Loss only]
    
    Final --> End([Backpropagation])
    Final2 --> End
    
    style Start fill:#e1f5e1
    style End fill:#ffe1e1
    style EncOut fill:#e1e5ff
    style FinalLogits fill:#ffe5e1
    style Input fill:#fff5e1
    style LogMel fill:#f0f0f0
    style E3Out fill:#e1e5ff
    style AttnOut fill:#ffe5f0
```

## Detailed Dimension Flow Table

| Step | Component | Input Dimension | Output Dimension | Notes |
|------|-----------|----------------|------------------|-------|
| 1 | **Audio Loading** | WAV file | `[time]` | 16kHz sampling rate |
| 2 | **Log-Mel Extraction** | `[time]` | `[T, 80]` | T ≈ time × 100 (10ms hop) |
| 3 | **CMVN** | `[T, 80]` | `[T, 80]` | Mean-variance normalization |
| 4 | **SpecAugment** | `[T, 80]` | `[T, 80]` | Only during training |
| 5 | **Batch Formation** | `[T, 80]` | `[B, T, 80]` | B = batch size |
| | **ENCODER** | | | |
| 6 | Bottom BiLSTM | `[B, T, 80]` | `[B, T, 256]` | 128×2 (bidirectional) |
| 7 | Pyramid Reduction 1 | `[B, T, 256]` | `[B, T/2, 512]` | Concatenate pairs |
| 8 | Pyramid BiLSTM 1 | `[B, T/2, 512]` | `[B, T/2, 256]` | 128×2 |
| 9 | Pyramid Reduction 2 | `[B, T/2, 256]` | `[B, T/4, 512]` | Concatenate pairs |
| 10 | Pyramid BiLSTM 2 | `[B, T/4, 512]` | `[B, T/4, 256]` | 128×2 |
| 11 | **Encoder Output** | - | `[B, T', 256]` | T' = T/4 |
| | **CTC HEAD (Optional)** | | | |
| 12 | CTC Linear | `[B, T', 256]` | `[B, T', V]` | V = vocab_size |
| | **DECODER** | | | |
| 13 | Embedding | `[B]` indices | `[B, 64]` | Token embeddings |
| 14 | Concatenate | `[B, 64]` + `[B, 256]` | `[B, 320]` | Embed + prev context |
| 15 | LSTM Cell | `[B, 320]` | `[B, 256]` | Decoder hidden state |
| 16 | **Location Attention** | | | |
| 16a | - Location Conv | `[B, T']` | `[B, T', 10]` | 1D conv on prev attention |
| 16b | - Query Projection | `[B, 256]` | `[B, 1, 128]` | Decoder state |
| 16c | - Key Projection | `[B, T', 256]` | `[B, T', 128]` | Encoder outputs |
| 16d | - Location Projection | `[B, T', 10]` | `[B, T', 128]` | Location features |
| 16e | - Energy Computation | All above | `[B, T', 128]` | Sum of projections |
| 16f | - Energy Linear | `[B, T', 128]` | `[B, T']` | Scalar energy |
| 16g | - Softmax | `[B, T']` | `[B, T']` | Attention weights |
| 16h | - Weighted Sum | `[B, T', 256]` × `[B, T']` | `[B, 256]` | Context vector |
| 17 | Projection | `[B, 256]` | `[B, V]` | Vocabulary logits |
| 18 | **Loop L times** | - | `[B, L, V]` | L = target length |
| | **LOSS** | | | |
| 19 | Cross-Entropy | `[B, L, V]` vs `[B, L]` | scalar | Attention loss |
| 20 | CTC Loss | `[B, T', V]` vs `[B, L]` | scalar | Optional |
| 21 | **Combined Loss** | - | scalar | λ×CTC + (1-λ)×CE |

## Example with Concrete Numbers

Assume:
- **Batch size (B)**: 8
- **Audio duration**: 3 seconds
- **Vocabulary size (V)**: 100 tokens
- **Target length (L)**: 20 characters

| Layer | Dimension | Calculation |
|-------|-----------|-------------|
| Audio waveform | `[48000]` | 3s × 16000 Hz |
| Log-Mel features | `[300, 80]` | 3s × 100 frames/sec |
| Batched input | `[8, 300, 80]` | Batch of 8 |
| After Bottom BiLSTM | `[8, 300, 256]` | BiLSTM hidden 128×2 |
| After Pyramid 1 | `[8, 150, 256]` | Time reduced by 2 |
| After Pyramid 2 | `[8, 75, 256]` | Time reduced by 4 total |
| **Encoder output** | `[8, 75, 256]` | Ready for decoder |
| CTC logits | `[8, 75, 100]` | If using CTC |
| Decoder per step | `[8, 100]` | Per character prediction |
| **Decoder output** | `[8, 20, 100]` | All 20 characters |

## Key Dimension Transformations

### Time Reduction (Pyramidal)
```
T (original frames)
  ↓ Bottom BiLSTM
T (same)
  ↓ Pyramid Level 1 (concat pairs)
T/2
  ↓ Pyramid Level 2 (concat pairs)
T/4 (final encoder time)
```

### Channel/Feature Dimension Expansion
```
80 (log-mel)
  ↓ BiLSTM (hidden=128, bidirectional)
256 (128 × 2)
  ↓ Pyramid concat
512 (256 × 2)
  ↓ BiLSTM
256 (128 × 2)
  ↓ Final
256 (encoder dimension)
```

### Decoder Input Feeding
```
Step t-1:
  Embedding[64] + Context[256] = [320]
    ↓ LSTM
  Hidden[256]
    ↓ Attention
  New Context[256] → used in step t
```

## Memory Requirements Estimation

For **batch_size=8**, **max_time=300 frames**, **vocab_size=100**:

| Component | Size | Calculation |
|-----------|------|-------------|
| Input features | 192 KB | 8 × 300 × 80 × 4 bytes |
| Encoder outputs | 615 KB | 8 × 75 × 256 × 4 bytes |
| Decoder logits | 64 KB | 8 × 20 × 100 × 4 bytes |
| Attention weights | 4.8 KB | 8 × 75 × 4 bytes |
| **Model parameters** | ~5-10 MB | Depends on config |

## Configuration → Dimensions Mapping

```python
Config Parameters          →  Resulting Dimensions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
n_mels = 80               →  Feature dim: [T, 80]
encoder_hidden_size = 128 →  Encoder out: [T', 256]
encoder_num_layers = 3    →  3 BiLSTM layers (1 bottom + 2 pyramid)
pyramid_levels = 2        →  Time reduction: T' = T/4
decoder_dim = 256         →  LSTM hidden: [B, 256]
attention_dim = 128       →  Attention space: [B, T', 128]
embedding_dim = 64        →  Token embed: [B, 64]
vocab_size = V            →  Output logits: [B, L, V]
```
