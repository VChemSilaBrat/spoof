import torch
import numpy as np
from tqdm import tqdm
from src.metrics.eer_utils import compute_eer


class ASVspoofEvaluator:
    """Evaluator for ASVspoof models"""
    
    def __init__(self, device):
        self.device = device
    
    @torch.no_grad()
    def evaluate(self, model, eval_loader):
        """
        Evaluate model on eval set
        
        Returns:
            eer: Equal Error Rate
            scores_dict: Dictionary mapping file_name to scores
        """
        model.eval()
        
        all_labels = []
        all_scores = []
        scores_dict = {}
        
        for batch in tqdm(eval_loader, desc='Evaluating'):
            features = batch['features'].to(self.device)
            labels = batch['label']
            file_names = batch['file_name']
            
            # Forward pass
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            
            # Get spoof probability (class 1)
            spoof_probs = probs[:, 1].cpu().numpy()
            
            all_labels.extend(labels.numpy())
            all_scores.extend(spoof_probs)
            
            # Store scores per file
            for fname, score in zip(file_names, spoof_probs):
                scores_dict[fname] = float(score)
        
        # Compute EER
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        eer, threshold = compute_eer(all_labels, all_scores)
        
        return eer, scores_dict
