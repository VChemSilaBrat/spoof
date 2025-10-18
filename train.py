#!/usr/bin/env python3
"""
Basic training script
"""
import torch
from torch.utils.data import DataLoader

from src.model.lightcnn_original import LightCNNOriginal
from src.datasets.asvspoof_dataset import ASVspoofDataset
from src.trainer.asvspoof_trainer import ASVspoofTrainer
from src.utils import set_random_seed


def main():
    # Setup
    set_random_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model
    model = LightCNNOriginal(
        input_shape=(1, 863, 600),
        num_classes=2,
        dropout_prob=0.75
    ).to(device)
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    # TODO: Add data loading
    print("Dataset loading not implemented yet...")


if __name__ == "__main__":
    main()
