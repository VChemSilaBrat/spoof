import torch
from tqdm import tqdm


class ASVspoofTrainer:
    """Trainer for ASVspoof models"""
    
    def __init__(self, model, train_loader, eval_loader, criterion, 
                 optimizer, scheduler, device, config):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        self.best_eer = float('inf')
        self.epoch = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for batch in pbar:
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs):
        """Train for multiple epochs"""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # Train
            train_loss, train_acc = self.train_epoch()
            print(f"Epoch {self.epoch}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        print("Training completed!")
