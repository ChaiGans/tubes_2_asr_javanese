# Javanese ASR Class Diagram

## Complete System Architecture

```mermaid
classDiagram
    %% Configuration
    class Config {
        +str audio_dir
        +str transcript_file
        +str vocab_path
        +int batch_size
        +int num_epochs
        +float learning_rate
        +float grad_clip_norm
        +float teacher_forcing_ratio
        +int input_dim
        +int encoder_hidden_size
        +int encoder_num_layers
        +int decoder_dim
        +int attention_dim
        +int embedding_dim
        +float dropout
        +bool use_ctc
        +float ctc_weight
        +int sample_rate
        +int n_mels
        +float win_length_ms
        +float hop_length_ms
        +bool apply_cmvn
        +bool apply_spec_augment
        +bool speed_perturb
        +float val_split
        +str checkpoint_dir
        +int max_decode_len
        +int beam_size
        +str device
        +int seed
    }

    %% Vocabulary
    class Vocabulary {
        +str PAD
        +str SOS
        +str EOS
        +str UNK
        +str BLANK
        +List~str~ special_tokens
        +Dict~str,int~ char2idx
        +Dict~int,str~ idx2char
        +int pad_idx
        +int sos_idx
        +int eos_idx
        +int unk_idx
        +int blank_idx
        +build_from_transcripts(transcripts)
        +encode(text, add_sos, add_eos) List~int~
        +decode(indices, remove_special) str
        +save(filepath)
        +load(filepath) Vocabulary
        +__len__() int
    }

    %% Feature Extraction
    class LogMelFeatureExtractor {
        +int sample_rate
        +int n_mels
        +int win_length
        +int hop_length
        +int n_fft
        +MelSpectrogram mel_transform
        +__call__(waveform) Tensor
    }

    class CMVN {
        +bool norm_means
        +bool norm_vars
        +__call__(features, lengths) Tensor
    }

    class SpecAugment {
        +int freq_mask_param
        +int time_mask_param
        +int num_freq_masks
        +int num_time_masks
        +float mask_value
        +forward(features) Tensor
    }

    %% Dataset
    class JavaneseASRDataset {
        +str audio_dir
        +Path audio_path
        +Vocabulary vocab
        +LogMelFeatureExtractor feature_extractor
        +CMVN cmvn
        +SpecAugment spec_augment
        +bool apply_spec_augment
        +bool speed_perturb
        +List~Dict~ data
        +_load_transcripts(transcript_file)
        +_validate_audio_files(data)
        +__len__() int
        +__getitem__(idx) Tuple
        +collate_fn(batch) Dict
    }

    %% Model Components
    class PyramidalBiLSTMEncoder {
        +int input_dim
        +int hidden_size
        +int num_layers
        +float dropout
        +int pyramid_levels
        +LSTM bottom_lstm
        +ModuleList pyramid_lstms
        +_apply_pyramidal_reduction(x, lengths) Tuple
        +forward(x, lengths) Tuple
    }

    class LocationSensitiveAttention {
        +int encoder_dim
        +int decoder_dim
        +int attention_dim
        +int num_filters
        +int kernel_size
        +Linear query_projection
        +Linear key_projection
        +Linear value_projection
        +Conv1d location_conv
        +Linear location_projection
        +Linear energy_projection
        +forward(decoder_state, encoder_outputs, prev_attention_weights, encoder_mask) Tuple
    }

    class DecoderWithAttention {
        +int vocab_size
        +int embedding_dim
        +int encoder_dim
        +int decoder_dim
        +int attention_dim
        +Embedding embedding
        +LSTMCell decoder_cell
        +LocationSensitiveAttention attention
        +Linear projection
        +Dropout dropout
        +forward_step(prev_token, prev_context, decoder_state, encoder_outputs, prev_attention_weights, encoder_mask) Tuple
        +forward(encoder_outputs, encoder_lengths, targets, teacher_forcing_ratio) Tuple
        +_create_mask(max_len, lengths) Tensor
    }

    class Seq2SeqASR {
        +int vocab_size
        +bool use_ctc
        +PyramidalBiLSTMEncoder encoder
        +DecoderWithAttention decoder
        +Linear ctc_head
        +forward(features, feature_lengths, targets, teacher_forcing_ratio) Tuple
        +compute_loss(logits, targets, target_lengths, ctc_logits, encoder_lengths, pad_idx, blank_idx) Tensor
    }

    %% Decoders
    class GreedyDecoder {
        +Seq2SeqASR model
        +Vocabulary vocab
        +int max_len
        +str device
        +decode(features, feature_lengths) List~str~
        +_create_mask(max_len, lengths) Tensor
    }

    class BeamSearchDecoder {
        +Seq2SeqASR model
        +Vocabulary vocab
        +int beam_size
        +int max_len
        +str device
        +float length_penalty
        +decode(features, feature_lengths) List~str~
        +_beam_search_single(features, feature_lengths) str
        +_create_mask(max_len, lengths) Tensor
    }

    %% Experiment Components
    class ExperimentTracker {
        +Path results_dir
        +Path results_file
        +List~Dict~ all_results
        +save_experiment(experiment_config, results) int
        +get_summary() str
    }

    class ExperimentRunner {
        +Config base_config
        +ExperimentTracker tracker
        +prepare_data(config) Tuple
        +run_single_experiment(experiment_name, config_overrides) Dict
    }

    %% Relationships - Composition
    JavaneseASRDataset *-- Vocabulary : uses
    JavaneseASRDataset *-- LogMelFeatureExtractor : uses
    JavaneseASRDataset *-- CMVN : has
    JavaneseASRDataset *-- SpecAugment : has

    Seq2SeqASR *-- PyramidalBiLSTMEncoder : has
    Seq2SeqASR *-- DecoderWithAttention : has
    
    DecoderWithAttention *-- LocationSensitiveAttention : has

    GreedyDecoder o-- Seq2SeqASR : uses
    GreedyDecoder o-- Vocabulary : uses
    
    BeamSearchDecoder o-- Seq2SeqASR : uses
    BeamSearchDecoder o-- Vocabulary : uses

    ExperimentRunner *-- Config : uses
    ExperimentRunner *-- ExperimentTracker : has
    ExperimentRunner ..> JavaneseASRDataset : creates
    ExperimentRunner ..> Seq2SeqASR : creates
    ExperimentRunner ..> GreedyDecoder : creates
    ExperimentRunner ..> BeamSearchDecoder : creates

    %% Data Flow
    Config ..> JavaneseASRDataset : configures
    Config ..> Seq2SeqASR : configures
    Config ..> LogMelFeatureExtractor : configures
```

## Simplified View: Core Model Architecture

```mermaid
classDiagram
    class Seq2SeqASR {
        +PyramidalBiLSTMEncoder encoder
        +DecoderWithAttention decoder
        +Linear ctc_head
        +bool use_ctc
        +forward()
        +compute_loss()
    }

    class PyramidalBiLSTMEncoder {
        +LSTM bottom_lstm
        +ModuleList pyramid_lstms
        +forward()
    }

    class DecoderWithAttention {
        +Embedding embedding
        +LSTMCell decoder_cell
        +LocationSensitiveAttention attention
        +Linear projection
        +forward()
    }

    class LocationSensitiveAttention {
        +Conv1d location_conv
        +Linear projections
        +forward()
    }

    Seq2SeqASR *-- PyramidalBiLSTMEncoder
    Seq2SeqASR *-- DecoderWithAttention
    DecoderWithAttention *-- LocationSensitiveAttention
```

## Data Pipeline Flow

```mermaid
classDiagram
    class JavaneseASRDataset {
        +load audio files
        +extract features
        +apply augmentation
        +encode transcripts
    }

    class LogMelFeatureExtractor {
        +extract log-mel features
    }

    class CMVN {
        +normalize features
    }

    class SpecAugment {
        +augment features
    }

    class Vocabulary {
        +encode text
        +decode indices
    }

    JavaneseASRDataset --> LogMelFeatureExtractor : 1. extract
    JavaneseASRDataset --> CMVN : 2. normalize
    JavaneseASRDataset --> SpecAugment : 3. augment
    JavaneseASRDataset --> Vocabulary : 4. encode
```

## Training Flow

```mermaid
classDiagram
    class Config {
        +training params
        +model params
    }

    class JavaneseASRDataset {
        +load data
    }

    class Seq2SeqASR {
        +forward()
        +compute_loss()
    }

    class Optimizer {
        +update weights
    }

    class ExperimentRunner {
        +run experiments
    }

    ExperimentRunner --> Config : reads
    ExperimentRunner --> JavaneseASRDataset : creates
    ExperimentRunner --> Seq2SeqASR : creates
    ExperimentRunner --> Optimizer : creates
    
    JavaneseASRDataset --> Seq2SeqASR : feeds data
    Seq2SeqASR --> Optimizer : gradients
```

## Inference Flow

```mermaid
classDiagram
    class Seq2SeqASR {
        +encoder
        +decoder
    }

    class GreedyDecoder {
        +decode with greedy search
    }

    class BeamSearchDecoder {
        +decode with beam search
    }

    class Vocabulary {
        +decode indices to text
    }

    GreedyDecoder --> Seq2SeqASR : uses model
    BeamSearchDecoder --> Seq2SeqASR : uses model
    GreedyDecoder --> Vocabulary : decodes with
    BeamSearchDecoder --> Vocabulary : decodes with
```

## Key Attributes Summary

### Model Hierarchy
```
Seq2SeqASR
├── encoder: PyramidalBiLSTMEncoder
│   ├── input_dim: 80 (mel features)
│   ├── hidden_size: 128
│   └── num_layers: 3
├── decoder: DecoderWithAttention
│   ├── vocab_size: from Vocabulary
│   ├── embedding_dim: 64
│   ├── decoder_dim: 256
│   ├── attention: LocationSensitiveAttention
│   │   ├── encoder_dim: 256
│   │   ├── decoder_dim: 256
│   │   └── attention_dim: 128
│   └── projection: Linear(decoder_dim, vocab_size)
└── ctc_head: Linear(hidden_size*2, vocab_size) [optional]
```

### Data Flow
```
Audio File (.wav)
    ↓
LogMelFeatureExtractor → [time, 80]
    ↓
CMVN (normalize) → [time, 80]
    ↓
SpecAugment (augment) → [time, 80]
    ↓
PyramidalBiLSTMEncoder → [reduced_time, 256]
    ↓
DecoderWithAttention → [target_len, vocab_size]
    ↓
Loss Computation (CrossEntropy + CTC)
```

### Configuration Dependencies
```
Config
├── Defines: audio_dir, transcript_file, vocab_path
├── Used by: JavaneseASRDataset
├── Defines: encoder_hidden_size, decoder_dim, etc.
├── Used by: Seq2SeqASR
├── Defines: sample_rate, n_mels
└── Used by: LogMelFeatureExtractor
```
