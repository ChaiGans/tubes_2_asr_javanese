"""
Feature Extraction for ASR: Log-Mel Filterbanks, CMVN, SpecAugment
"""

import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
from typing import Optional, Tuple


class LogMelFeatureExtractor:
    """
    Extract 80-dim log-mel filterbank features from raw audio.
    
    Args:
        sample_rate: Audio sample rate (16000 Hz)
        n_mels: Number of mel filterbanks (80)
        win_length: Window length in ms (25 ms)
        hop_length: Hop length in ms (10 ms)
        n_fft: FFT size (default 512 for 16kHz)
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        win_length_ms: float = 25.0,
        hop_length_ms: float = 10.0,
        n_fft: int = 512
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Convert ms to samples
        self.win_length = int(sample_rate * win_length_ms / 1000)
        self.hop_length = int(sample_rate * hop_length_ms / 1000)
        self.n_fft = n_fft
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=n_mels,
            f_min=0.0,
            f_max=sample_rate / 2,
            power=2.0,
            normalized=False
        )
        
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract log-mel features from waveform.
        
        Args:
            waveform: [channels, time] or [time]
        
        Returns:
            log_mel: [time, n_mels] - log-mel features
        """
        # Ensure waveform is 2D: [channels, time]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        # Compute mel spectrogram: [channels, n_mels, time]
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale (add small epsilon for numerical stability)
        log_mel = torch.log(mel_spec + 1e-9)
        
        # Take first channel and transpose to [time, n_mels]
        log_mel = log_mel[0].transpose(0, 1)
        
        return log_mel

class CMVN:
    """
    Cepstral Mean and Variance Normalization.
    Can be applied per-utterance or globally.
    """
    def __init__(self, norm_means: bool = True, norm_vars: bool = True):
        self.norm_means = norm_means
        self.norm_vars = norm_vars
    
    def __call__(self, features: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply CMVN normalization.
        
        Args:
            features: [batch, time, feat_dim] or [time, feat_dim]
            lengths: [batch] - if provided, normalize per utterance; else global
        
        Returns:
            normalized_features: Same shape as input
        """
        is_batched = features.ndim == 3 
        
        if not is_batched:
            features = features.unsqueeze(0)  # [1, time, feat_dim]

        if self.norm_means:
            mean = features.mean(dim=(0, 1), keepdim=True)
            features = features - mean
        
        if self.norm_vars:
            std = features.std(dim=(0, 1), keepdim=True) + 1e-9
            features = features / std
        
        if not is_batched:
            features = features.squeeze(0)
        
        return features

class SpecAugment(nn.Module):
    """
    SpecAugment: Data augmentation for speech.
    
    Applies frequency masking and time masking to log-mel features.
    Reference: https://arxiv.org/abs/1904.08779
    
    Args:
        freq_mask_param: Maximum frequency mask width (F in paper)
        time_mask_param: Maximum time mask width (T in paper)
        num_freq_masks: Number of frequency masks to apply
        num_time_masks: Number of time masks to apply
        mask_value: Value to fill masked regions (0.0 for log-mel)
    """
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
        mask_value: float = 0.0
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value
        
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to features.
        
        Args:
            features: [batch, time, freq] or [time, freq]
        
        Returns:
            augmented_features: Same shape as input
        """
        is_batched = features.ndim == 3
        
        if not is_batched:
            features = features.unsqueeze(0)  # [1, time, freq]
        
        # Transpose to [batch, freq, time] for torchaudio transforms
        features = features.transpose(1, 2)
        
        # Apply frequency masks
        for _ in range(self.num_freq_masks):
            features = self.freq_mask(features)
        
        # Apply time masks
        for _ in range(self.num_time_masks):
            features = self.time_mask(features)
        
        # Transpose back to [batch, time, freq]
        features = features.transpose(1, 2)
        
        if not is_batched:
            features = features.squeeze(0)
        
        return features


def load_audio(filepath, target_sr=16000):
    # Read audio using soundfile (no torchcodec)
    waveform_np, sr = sf.read(filepath, dtype="float32")  # shape: (num_samples,) or (num_samples, channels)

    # Convert to torch tensor and ensure shape (channels, num_samples)
    waveform = torch.from_numpy(waveform_np)
    if waveform.ndim == 1:
        # mono
        waveform = waveform.unsqueeze(0)  # (1, num_samples)
    elif waveform.ndim == 2:
        # (num_samples, channels) -> (channels, num_samples)
        waveform = waveform.transpose(0, 1)

    # Resample if needed
    if sr != target_sr:
        waveform = F.resample(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr

if __name__ == "__main__":
    print("Testing feature extraction...")
    
    # Create dummy audio
    sample_rate = 16000
    duration = 3.0  # seconds
    waveform = torch.randn(int(sample_rate * duration))
    
    # Extract log-mel features
    extractor = LogMelFeatureExtractor(sample_rate=sample_rate, n_mels=80)
    log_mel = extractor(waveform)
    print(f"Log-mel features shape: {log_mel.shape}")  # [time_frames, 80]
    
    # Apply CMVN
    cmvn = CMVN(norm_means=True, norm_vars=True)
    normalized = cmvn(log_mel)
    print(f"After CMVN - mean: {normalized.mean():.4f}, std: {normalized.std():.4f}")
    
    # Apply SpecAugment
    spec_aug = SpecAugment(freq_mask_param=27, time_mask_param=100)
    augmented = spec_aug(normalized.unsqueeze(0)).squeeze(0)
    print(f"After SpecAugment shape: {augmented.shape}")
    
    # Test batched
    batch = torch.stack([log_mel, log_mel], dim=0)
    batch_normalized = cmvn(batch)
    print(f"Batch CMVN shape: {batch_normalized.shape}")
    
    print("Feature extraction test passed!")
