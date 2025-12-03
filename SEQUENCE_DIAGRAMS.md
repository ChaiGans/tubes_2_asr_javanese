# Javanese ASR - Sequence Diagrams

## Training Sequence

```mermaid
sequenceDiagram
    participant Main
    participant Config
    participant Vocab
    participant Dataset as JavaneseASRDataset
    participant FE as LogMelFeatureExtractor
    participant Model as Seq2SeqASR
    participant Encoder as PyramidalBiLSTMEncoder
    participant Decoder as DecoderWithAttention
    participant Optimizer

    Main->>Config: load configuration
    Main->>Vocab: load vocabulary
    Main->>FE: create feature extractor
    Main->>Dataset: create(audio_dir, vocab, FE)
    Dataset->>Dataset: load transcripts
    Dataset->>Dataset: validate audio files
    
    Main->>Model: create(vocab_size, config)
    Model->>Encoder: initialize encoder
    Model->>Decoder: initialize decoder
    
    Main->>Optimizer: create(model.parameters)
    
    loop For each epoch
        loop For each batch
            Dataset->>FE: extract features(audio)
            FE-->>Dataset: log-mel features [time, 80]
            Dataset->>Dataset: apply CMVN
            Dataset->>Dataset: apply SpecAugment
            Dataset->>Vocab: encode(transcript)
            Vocab-->>Dataset: target_indices
            Dataset-->>Main: batch{features, targets, lengths}
            
            Main->>Model: forward(features, targets)
            Model->>Encoder: forward(features, lengths)
            Encoder-->>Model: encoder_outputs [batch, T', 256]
            Model->>Decoder: forward(encoder_outputs, targets)
            Decoder-->>Model: logits [batch, L, vocab_size]
            Model->>Model: compute_loss(logits, targets)
            Model-->>Main: loss
            
            Main->>Optimizer: zero_grad()
            Main->>Model: loss.backward()
            Main->>Optimizer: step()
        end
    end
```

## Inference Sequence (Greedy Decoding)

```mermaid
sequenceDiagram
    participant Main
    participant Decoder as GreedyDecoder
    participant Model as Seq2SeqASR
    participant Encoder as PyramidalBiLSTMEncoder
    participant DecoderModel as DecoderWithAttention
    participant Attention as LocationSensitiveAttention
    participant Vocab

    Main->>Decoder: decode(features, lengths)
    Decoder->>Model: encoder.forward(features, lengths)
    Model->>Encoder: forward(features, lengths)
    Encoder-->>Model: encoder_outputs [batch, T', 256]
    Model-->>Decoder: encoder_outputs
    
    Decoder->>Decoder: init decoder state
    Decoder->>Decoder: prev_token = <sos>
    
    loop Until <eos> or max_len
        Decoder->>DecoderModel: forward_step(prev_token, prev_context, state, encoder_outputs)
        DecoderModel->>DecoderModel: embed(prev_token)
        DecoderModel->>DecoderModel: concat[embedding, prev_context]
        DecoderModel->>DecoderModel: LSTM_cell(input, state)
        DecoderModel->>Attention: forward(decoder_state, encoder_outputs, prev_attn_weights)
        Attention-->>DecoderModel: context, attention_weights
        DecoderModel->>DecoderModel: project(context)
        DecoderModel-->>Decoder: logits, new_state, context, attn_weights
        
        Decoder->>Decoder: prev_token = argmax(logits)
        Decoder->>Decoder: append to sequence
    end
    
    Decoder->>Vocab: decode(sequence)
    Vocab-->>Decoder: text
    Decoder-->>Main: transcripts
```

## Inference Sequence (Beam Search)

```mermaid
sequenceDiagram
    participant Main
    participant BeamDecoder as BeamSearchDecoder
    participant Model as Seq2SeqASR
    participant Encoder
    participant Decoder as DecoderWithAttention
    participant Vocab

    Main->>BeamDecoder: decode(features, lengths)
    BeamDecoder->>Model: encoder.forward(features)
    Model->>Encoder: forward(features, lengths)
    Encoder-->>Model: encoder_outputs
    Model-->>BeamDecoder: encoder_outputs
    
    BeamDecoder->>BeamDecoder: init beam = [(<sos>, 0.0, state)]
    
    loop For max_len steps
        loop For each hypothesis in beam
            BeamDecoder->>Decoder: forward_step(token, context, state, encoder_outputs)
            Decoder-->>BeamDecoder: logits, new_state, context, attn_weights
            BeamDecoder->>BeamDecoder: compute probabilities
            BeamDecoder->>BeamDecoder: expand hypothesis
        end
        
        BeamDecoder->>BeamDecoder: select top K candidates
        BeamDecoder->>BeamDecoder: update beam
        
        alt All beams end with <eos>
            BeamDecoder->>BeamDecoder: break
        end
    end
    
    BeamDecoder->>BeamDecoder: select best hypothesis
    BeamDecoder->>Vocab: decode(best_sequence)
    Vocab-->>BeamDecoder: text
    BeamDecoder-->>Main: transcript
```

## Data Loading Sequence

```mermaid
sequenceDiagram
    participant Main
    participant Dataset as JavaneseASRDataset
    participant Vocab
    participant FE as LogMelFeatureExtractor
    participant CMVN
    participant SpecAug as SpecAugment
    participant Audio

    Main->>Dataset: __getitem__(idx)
    Dataset->>Dataset: get data[idx]
    Dataset->>Audio: load_audio(audio_path)
    Audio-->>Dataset: waveform [time]
    
    alt speed_perturb enabled
        Dataset->>Dataset: speed_perturb(waveform)
    end
    
    Dataset->>FE: __call__(waveform)
    FE->>FE: mel_transform(waveform)
    FE->>FE: log(mel_spec + epsilon)
    FE->>FE: transpose to [time, n_mels]
    FE-->>Dataset: log_mel features [time, 80]
    
    alt apply_cmvn enabled
        Dataset->>CMVN: __call__(features)
        CMVN->>CMVN: compute mean & std
        CMVN->>CMVN: normalize
        CMVN-->>Dataset: normalized features
    end
    
    alt apply_spec_augment enabled and training
        Dataset->>SpecAug: forward(features)
        SpecAug->>SpecAug: apply frequency masking
        SpecAug->>SpecAug: apply time masking
        SpecAug-->>Dataset: augmented features
    end
    
    Dataset->>Vocab: encode(transcript)
    Vocab-->>Dataset: target_indices [L]
    
    Dataset-->>Main: (features, targets, transcript, utt_id)
```

## Experiment Runner Sequence

```mermaid
sequenceDiagram
    participant Main
    participant Runner as ExperimentRunner
    participant Tracker as ExperimentTracker
    participant Config
    participant Dataset
    participant Model
    participant Train as train_one_epoch
    participant Validate as validate

    Main->>Runner: run_single_experiment(name, config_overrides)
    Runner->>Config: create config with overrides
    Runner->>Dataset: prepare_data(config)
    Dataset-->>Runner: train_loader, val_loader, vocab
    
    Runner->>Model: create Seq2SeqASR(config)
    Model-->>Runner: model
    
    Runner->>Runner: create optimizer
    
    loop For each epoch
        Runner->>Train: train_one_epoch(model, train_loader, optimizer)
        
        loop For each batch
            Train->>Model: forward(features, targets)
            Model-->>Train: loss
            Train->>Train: loss.backward()
            Train->>Train: clip_gradients()
            Train->>Train: optimizer.step()
        end
        
        Train-->>Runner: avg_train_loss
        
        Runner->>Validate: validate(model, val_loader)
        
        loop For each batch
            Validate->>Model: forward(features, targets)
            Model-->>Validate: loss, logits
            Validate->>Validate: decode predictions
            Validate->>Validate: compute CER
        end
        
        Validate-->>Runner: val_loss, val_cer
    end
    
    Runner->>Tracker: save_experiment(config, results)
    Tracker->>Tracker: assign experiment_id
    Tracker->>Tracker: save to JSON
    Tracker-->>Runner: experiment_id
    
    Runner-->>Main: results
```

## Model Forward Pass Details

```mermaid
sequenceDiagram
    participant Input
    participant Encoder as PyramidalBiLSTMEncoder
    participant LSTM
    participant Decoder as DecoderWithAttention
    participant Attention
    participant Output

    Input->>Encoder: features [B, T, 80]
    Encoder->>LSTM: bottom_lstm(features)
    LSTM-->>Encoder: outputs [B, T, hidden*2]
    
    loop For each pyramid level
        Encoder->>Encoder: reduce_time(outputs) [B, T/2, hidden*4]
        Encoder->>LSTM: pyramid_lstm(reduced)
        LSTM-->>Encoder: outputs [B, T/2, hidden*2]
    end
    
    Encoder-->>Input: encoder_outputs [B, T', 256]
    
    Input->>Decoder: forward(encoder_outputs, targets)
    Decoder->>Decoder: init state, context, attn_weights
    
    loop For each target position
        Decoder->>Decoder: embed(prev_token)
        Decoder->>Decoder: concat[embedding, prev_context]
        Decoder->>Decoder: lstm_cell(concat, state)
        
        Decoder->>Attention: forward(state, encoder_outputs, prev_attn)
        Attention->>Attention: location_conv(prev_attn)
        Attention->>Attention: compute energy
        Attention->>Attention: softmax(energy)
        Attention->>Attention: weighted sum
        Attention-->>Decoder: context, attn_weights
        
        Decoder->>Decoder: project(context)
        Decoder->>Decoder: store logits
    end
    
    Decoder-->>Output: logits [B, L, vocab_size]
```
