import os
import torch
from tqdm import tqdm
from .evaluator import ASVspoofEvaluator


class ASVspoofTrainer:
    """Trainer for ASVspoof models"""
    
    def __init__(self, model, train_loader, eval_loader, criterion, 
                 optimizer, scheduler, device, config, writer=None):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.writer = writer
        
        self.evaluator = ASVspoofEvaluator(device)
        self.best_eer = float('inf')
        self.epoch = 0
        self.step = 0
    
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
            
            # Log to Comet.ml
            if self.writer:
                self.writer.log_metric("batch_loss", loss.item(), step=self.step)
            self.step += 1
            
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
            
            # Log epoch metrics
            if self.writer:
                self.writer.log_metrics({
                    "epoch_loss": train_loss,
                    "epoch_accuracy": train_acc
                }, epoch=self.epoch)
            
            # Evaluate
            if self.eval_loader:
                eval_eer, _ = self.evaluator.evaluate(self.model, self.eval_loader)
                print(f"Evaluation EER: {eval_eer:.2f}%")
                
                if self.writer:
                    self.writer.log_metric("eval_eer", eval_eer, epoch=self.epoch)
                
                # Save best model
                if eval_eer < self.best_eer:
                    self.best_eer = eval_eer
                    print(f"New best EER: {eval_eer:.2f}%")
                    self.save_checkpoint('best_model.pth')
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning rate: {lr:.6f}")
                if self.writer:
                    self.writer.log_metric("learning_rate", lr, epoch=self.epoch)
        
        print("Training completed!")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        save_dir = self.config.get('save_dir', 'saved')
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_eer': self.best_eer
        }
        
        filepath = os.path.join(save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
