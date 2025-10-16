import torch
import torchaudio
import numpy as np


def load_audio(file_path, target_sr=16000):
    """Load audio file and resample if needed"""
    waveform, sr = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    return waveform, target_sr


def compute_stft(waveform, n_fft=1724, hop_length=172, win_length=1724):
    """
    Compute Short-Time Fourier Transform
    
    Args:
        waveform: Audio waveform tensor
        n_fft: FFT size
        hop_length: Hop length for STFT
        win_length: Window length
    
    Returns:
        STFT magnitude spectrogram
    """
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=None  # Return complex values
    )
    
    stft = stft_transform(waveform)
    
    # Get magnitude
    magnitude = torch.abs(stft)
    
    return magnitude


def pad_or_truncate(features, target_length=600):
    """
    Pad or truncate features to target length
    
    Args:
        features: Feature tensor (channels, freq, time)
        target_length: Target time dimension
    
    Returns:
        Padded or truncated features
    """
    current_length = features.shape[-1]
    
    if current_length < target_length:
        # Pad
        pad_length = target_length - current_length
        features = torch.nn.functional.pad(features, (0, pad_length))
    elif current_length > target_length:
        # Truncate
        features = features[..., :target_length]
    
    return features
