#!/usr/bin/env python3
"""
Evaluation script for trained ASVspoof models
"""
import os
import torch
import pandas as pd

from src.model.lightcnn_original import LightCNNOriginal
from src.datasets import ASVspoofDataset, get_eval_loader
from src.trainer.evaluator import ASVspoofEvaluator


def main():
    # Configuration
    model_path = "saved/best_model.pth"
    dataset_path = "LA"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = LightCNNOriginal(
        input_shape=(1, 863, 600),
        num_classes=2,
        dropout_prob=0.75
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("Model loaded successfully!")
    
    # Load evaluation dataset
    eval_dataset = ASVspoofDataset(
        data_dir=os.path.join(dataset_path, 'ASVspoof2019_LA_eval/flac'),
        protocol_file=os.path.join(dataset_path,
                                   'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt')
    )
    
    eval_loader = get_eval_loader(eval_dataset, batch_size=16, num_workers=4)
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Evaluate
    evaluator = ASVspoofEvaluator(device)
    eval_eer, scores_dict = evaluator.evaluate(model, eval_loader)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"{'='*50}")
    print(f"Equal Error Rate (EER): {eval_eer:.2f}%")
    print(f"{'='*50}\n")
    
    # Save predictions
    output_file = "eval_predictions.csv"
    df = pd.DataFrame([
        {'file_name': fname, 'score': score}
        for fname, score in scores_dict.items()
    ])
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    main()
