import os
import torch
from torch.utils.data import Dataset
from .data_utils import load_audio, compute_stft, pad_or_truncate


class ASVspoofDataset(Dataset):
    """ASVspoof 2019 LA dataset"""
    
    def __init__(self, data_dir, protocol_file, transform=None):
        """
        Args:
            data_dir: Path to audio files
            protocol_file: Path to protocol file
            transform: Optional transform to be applied
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Parse protocol file
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    speaker_id = parts[0]
                    file_name = parts[1]
                    label = parts[4] if len(parts) > 4 else parts[-1]
                    
                    # Convert label: bonafide=0, spoof=1
                    label_idx = 0 if label == 'bonafide' else 1
                    
                    self.samples.append({
                        'file_name': file_name,
                        'label': label_idx,
                        'speaker_id': speaker_id
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        audio_path = os.path.join(self.data_dir, f"{sample['file_name']}.flac")
        waveform, sr = load_audio(audio_path)
        
        # Compute STFT features
        features = compute_stft(waveform)
        
        # Pad or truncate to fixed length
        features = pad_or_truncate(features, target_length=600)
        
        # Apply transform if any
        if self.transform:
            features = self.transform(features)
        
        return {
            'features': features,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'file_name': sample['file_name']
        }
