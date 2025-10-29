#!/usr/bin/env python3
"""
Training script for ASVspoof LightCNN
"""
import os

# Import comet_ml BEFORE torch for proper logging
from src.logger.cometml import CometMLWriter

import torch
from src.model.lightcnn_original import LightCNNOriginal
from src.datasets import ASVspoofDataset, get_train_loader, get_eval_loader
from src.trainer.asvspoof_trainer import ASVspoofTrainer
from src.utils import set_random_seed


def main():
    # Configuration
    config = {
        'seed': 42,
        'epochs': 16,
        'batch_size': 16,
        'num_workers': 0,  # Set to 0 on Windows to avoid multiprocessing issues
        'learning_rate': 0.0005,
        'scheduler_gamma': 0.98,
        'dropout_prob': 0.75,
        'dataset_path': 'LA'
    }
    
    # Setup
    set_random_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Comet.ml writer
    writer = CometMLWriter(
        project_name="asvspoof-lightcnn",
        experiment_name="baseline-experiment"
    )
    writer.log_parameters(config)
    
    # Create datasets
    train_dataset = ASVspoofDataset(
        data_dir=os.path.join(config['dataset_path'], 'ASVspoof2019_LA_train/flac'),
        protocol_file=os.path.join(config['dataset_path'], 
                                   'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
    )
    
    eval_dataset = ASVspoofDataset(
        data_dir=os.path.join(config['dataset_path'], 'ASVspoof2019_LA_dev/flac'),
        protocol_file=os.path.join(config['dataset_path'],
                                   'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt')
    )
    
    # Create data loaders
    train_loader = get_train_loader(
        train_dataset, 
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    eval_loader = get_eval_loader(
        eval_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Model
    model = LightCNNOriginal(
        input_shape=(1, 863, 600),
        num_classes=2,
        dropout_prob=config['dropout_prob']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, 
        gamma=config['scheduler_gamma']
    )
    
    # Trainer
    trainer = ASVspoofTrainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        writer=writer
    )
    
    # Train
    trainer.train(num_epochs=config['epochs'])
    
    # End experiment
    writer.end()
    print("Training finished!")


if __name__ == "__main__":
    main()
