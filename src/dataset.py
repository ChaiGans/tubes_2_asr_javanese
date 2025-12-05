"""
Dataset class for Javanese ASR
Loads audio files and transcripts, applies feature extraction and augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random

from src.features import LogMelFeatureExtractor, CMVN, SpecAugment, load_audio
from src.vocab import Vocabulary
from src.text_preprocessor import TextPreprocessor


class JavaneseASRDataset(Dataset):
    """
    Dataset for Javanese ASR.
    
    Expects:
        - Audio files in audio_dir: SpeakerXX_m_nn_uttYY.wav (capital S)
        - Transcript file: CSV format with header "SentenceID,Transcript,Device"
    
    Args:
        audio_dir: Directory containing WAV files
        transcript_file: Path to transcript CSV file
        vocab: Vocabulary object
        feature_extractor: LogMelFeatureExtractor instance
        apply_cmvn: Whether to apply CMVN normalization
        apply_spec_augment: Whether to apply SpecAugment (training only)
        speed_perturb: Whether to apply speed perturbation (training only)
        preprocessor: Optional TextPreprocessor for transcript normalization
        use_preprocessing: Whether to apply text preprocessing (default: True)
    """
    def __init__(
        self,
        audio_dir: str,
        transcript_file: str,
        vocab: Vocabulary,
        feature_extractor: Optional[LogMelFeatureExtractor] = None,
        apply_cmvn: bool = True,
        apply_spec_augment: bool = False,
        speed_perturb: bool = False,
        preprocessor: Optional[TextPreprocessor] = None,
        use_preprocessing: bool = True
    ):
        self.audio_dir = Path(audio_dir)
        self.vocab = vocab
        self.apply_cmvn = apply_cmvn
        self.apply_spec_augment = apply_spec_augment
        self.speed_perturb = speed_perturb
        self.use_preprocessing = use_preprocessing
        
        # Text preprocessing
        if use_preprocessing and preprocessor is None:
            self.preprocessor = TextPreprocessor(seed=42, ampersand_replacement='random')
        else:
            self.preprocessor = preprocessor
        
        # Feature extraction
        if feature_extractor is None:
            self.feature_extractor = LogMelFeatureExtractor()
        else:
            self.feature_extractor = feature_extractor
        
        # CMVN and SpecAugment
        self.cmvn = CMVN(norm_means=True, norm_vars=True)
        self.spec_augment = SpecAugment(
            freq_mask_param=27,
            time_mask_param=100,
            num_freq_masks=1,
            num_time_masks=1
        )
        
        
        # Load transcript data
        self.data = self._load_transcripts(transcript_file)
        
        # Validate audio files and remove corrupted ones
        print(f"Validating audio files...")
        self.data = self._validate_audio_files(self.data)
        
        print(f"Loaded {len(self.data)} valid utterances from {transcript_file}")
    
    def _load_transcripts(self, transcript_file: str) -> List[Dict]:
        """Load transcript CSV file and create data list."""
        data = []
        
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                # Skip header row (first line)
                if line_idx == 0 and line.startswith('SentenceID'):
                    continue
                
                # Split by comma (CSV format)
                parts = line.split(',')
                if len(parts) >= 2:
                    utt_id = parts[0].strip()
                    transcript = parts[1].strip()
                    
                    # Apply preprocessing if enabled
                    if self.use_preprocessing and self.preprocessor:
                        transcript = self.preprocessor.preprocess(transcript)
                    
                    # Skip empty transcripts
                    if not transcript:
                        continue
                    
                    # Construct audio file path
                    # Try both capitalized and lowercase formats
                    # Format: speaker01_m_nn_utt01 -> Speaker01_m_nn_utt01.wav
                    audio_path_lower = self.audio_dir / f"{utt_id}.wav"
                    
                    # If the first letter of utt_id is lowercase 's', try capitalizing just the 'S'
                    if utt_id.startswith('speaker'):
                        audio_path_cap = self.audio_dir / f"S{utt_id[1:]}.wav"
                    else:
                        audio_path_cap = self.audio_dir / f"{utt_id.capitalize()}.wav"
                    
                    # Check which file exists
                    if audio_path_lower.exists():
                        audio_path = audio_path_lower
                    elif audio_path_cap.exists():
                        audio_path = audio_path_cap
                    else:
                        # Try one more format with dash instead of underscore for some speakers
                        audio_path_dash = self.audio_dir / f"{utt_id.replace('_', '-')}.wav"
                        if audio_path_dash.exists():
                            audio_path = audio_path_dash
                        else:
                            # Also try capitalized with dash
                            if utt_id.startswith('speaker'):
                                audio_path_cap_dash = self.audio_dir / f"S{utt_id[1:].replace('_', '-')}.wav"
                            else:
                                audio_path_cap_dash = self.audio_dir / f"{utt_id.capitalize().replace('_', '-')}.wav"
                            
                            if audio_path_cap_dash.exists():
                                audio_path = audio_path_cap_dash
                            else:
                                continue  # Skip if not found
                    
                    data.append({
                        'utt_id': utt_id,
                        'audio_path': str(audio_path),
                        'transcript': transcript
                    })
        
        return data
    
    def _validate_audio_files(self, data: List[Dict]) -> List[Dict]:
        """Validate audio files and remove corrupted ones."""
        import soundfile as sf
        valid_data = []
        corrupted_count = 0
        
        for item in data:
            try:
                # Try to read just the header to validate the file
                info = sf.info(item['audio_path'])
                valid_data.append(item)
            except Exception as e:
                corrupted_count += 1
                print(f"  Skipping corrupted file: {item['audio_path']}")
        
        if corrupted_count > 0:
            print(f"  Removed {corrupted_count} corrupted audio files")
        
        return valid_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """
        Get a single item.
        
        Returns:
            features: [time, n_mels] - log-mel features
            target_indices: [target_len] - encoded target sequence (with <sos>, <eos>)
            transcript: Original text
            utt_id: Utterance ID
        """
        # Try to load the item, if it fails (corrupted file), skip to next
        max_retries = 10
        for retry in range(max_retries):
            try:
                item = self.data[(idx + retry) % len(self.data)]
                
                # Load audio
                waveform, sr = load_audio(item['audio_path'], target_sr=16000)
                
                # Speed perturbation (training augmentation)
                if self.speed_perturb and random.random() < 0.5:
                    speed_factor = random.choice([0.9, 1.1])
                    try:
                        from features import speed_perturb
                        waveform = speed_perturb(waveform, speed_factor)
                    except:
                        pass  # Skip if speed perturb fails
                
                # Extract features
                features = self.feature_extractor(waveform)  # [time, n_mels]
                
                # Apply CMVN (per-utterance)
                if self.apply_cmvn:
                    features = self.cmvn(features)
                
                # Apply SpecAugment (training augmentation)
                if self.apply_spec_augment:
                    features = self.spec_augment(features)
                
                # Encode transcript
                target_indices = torch.tensor(
                    self.vocab.encode(item['transcript'], add_sos=True, add_eos=True),
                    dtype=torch.long
                )
                
                return features, target_indices, item['transcript'], item['utt_id']
            
            except Exception as e:
                if retry == 0:  # Only print warning on first error for this idx
                    print(f"\nWarning: Error loading {item['audio_path']}: {e}")
                    print(f"  Skipping to next item...")
                continue
        
        # If all retries failed, raise error
        raise RuntimeError(f"Failed to load any valid audio after {max_retries} attempts")


def collate_fn(batch: List[Tuple]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    Pads features and targets to the same length within a batch.
    
    Args:
        batch: List of (features, target_indices, transcript, utt_id)
    
    Returns:
        Dictionary with:
            - features: [batch, max_time, n_mels]
            - feature_lengths: [batch]
            - targets: [batch, max_target_len]
            - target_lengths: [batch]
            - transcripts: List of strings
            - utt_ids: List of strings
    """
    features_list = [item[0] for item in batch]
    targets_list = [item[1] for item in batch]
    transcripts = [item[2] for item in batch]
    utt_ids = [item[3] for item in batch]
    
    # Get lengths
    feature_lengths = torch.tensor([f.size(0) for f in features_list], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in targets_list], dtype=torch.long)
    
    # Pad features to max length in batch
    max_feature_len = feature_lengths.max().item()
    n_mels = features_list[0].size(1)
    
    padded_features = torch.zeros(len(batch), max_feature_len, n_mels)
    for i, feat in enumerate(features_list):
        length = feat.size(0)
        padded_features[i, :length] = feat
    
    # Pad targets to max length in batch
    max_target_len = target_lengths.max().item()
    padded_targets = torch.zeros(len(batch), max_target_len, dtype=torch.long)
    
    for i, target in enumerate(targets_list):
        length = target.size(0)
        padded_targets[i, :length] = target
    
    return {
        'features': padded_features,
        'feature_lengths': feature_lengths,
        'targets': padded_targets,
        'target_lengths': target_lengths,
        'transcripts': transcripts,
        'utt_ids': utt_ids
    }


if __name__ == "__main__":
    print("Testing Dataset...")
    
    # This is a test with dummy data
    # In real usage, you would have actual audio files and transcripts
    
    from vocab import Vocabulary
    
    # Create dummy vocabulary
    vocab = Vocabulary()
    vocab.build_from_transcripts([
        "aku libur sedina",
        "wong jawa seneng"
    ])
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Note: This test requires actual audio files to run
    # You can test with your actual data by uncommenting below:
    """
    dataset = JavaneseASRDataset(
        audio_dir="audio_input",
        transcript_file="transcripts.csv",
        vocab=vocab,
        apply_cmvn=True,
        apply_spec_augment=True,
        speed_perturb=True
    )
    
    # Test single item
    features, target, transcript, utt_id = dataset[0]
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Transcript: {transcript}")
    print(f"Utterance ID: {utt_id}")
    
    # Test DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch features shape: {batch['features'].shape}")
    print(f"Batch targets shape: {batch['targets'].shape}")
    print(f"Feature lengths: {batch['feature_lengths']}")
    print(f"Target lengths: {batch['target_lengths']}")
    """
    
    print("\nDataset class created successfully!")
    print("To test with actual data, provide audio files and transcripts.")
