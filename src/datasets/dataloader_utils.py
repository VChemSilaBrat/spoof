import torch
from torch.utils.data import DataLoader


def collate_fn(batch):
    """Custom collate function for batching"""
    features = torch.stack([item['features'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    file_names = [item['file_name'] for item in batch]
    
    return {
        'features': features,
        'label': labels,
        'file_name': file_names
    }


def get_train_loader(dataset, batch_size=16, num_workers=4):
    """Create training data loader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def get_eval_loader(dataset, batch_size=16, num_workers=4):
    """Create evaluation data loader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
