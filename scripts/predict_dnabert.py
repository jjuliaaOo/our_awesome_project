import argparse
import pandas as pd
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.promoter_classifier.transformers.dnabert_model import DNABERTModel

def main():
    parser = argparse.ArgumentParser(description='Predict with DNABERT model')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input_csv', type=str, required=True, help='Input CSV with sequences')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV for predictions')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for prediction')
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_dir}")
    print(f"Input data: {args.input_csv}")
    print(f"Output: {args.output_csv}")
    
    # Load model
    dnabert = DNABERTModel()
    dnabert.load_from_dir(args.model_dir)
    
    # Load data
    df = pd.read_csv(args.input_csv)
    sequences = df['sequence'].tolist()
    
    print(f"Predicting for {len(sequences)} sequences...")
    
    # Predict in batches
    all_predictions = []
    all_probabilities = []
    
    for i in range(0, len(sequences), args.batch_size):
        batch_sequences = sequences[i:i + args.batch_size]
        
        for seq in batch_sequences:
            result = dnabert.predict_single(seq)
            all_predictions.append(result['prediction'])
            all_probabilities.append(result['probabilities'][1])  # Probability of class 1
    
    # Add predictions to dataframe
    df['dnabert_prediction'] = all_predictions
    df['dnabert_probability'] = all_probabilities
    
    # Save results
    df.to_csv(args.output_csv, index=False)
    
    print(f"Predictions saved to: {args.output_csv}")
    print(f"Class distribution: {df['dnabert_prediction'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
