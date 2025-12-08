import argparse
import torch
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.promoter_classifier.transformers.dnabert_model import DNABERTModel, DNABERTConfig
from src.promoter_classifier.transformers.dnabert_dataset import PromoterDNABERTDataset, DNABERTDataManager
from src.promoter_classifier.transformers.dnabert_trainer import DNABERTTrainer

def main():
    parser = argparse.ArgumentParser(description='Train DNABERT model')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--output_dir', type=str, default='./models/dnabert', help='Output directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Training DNABERT model with:")
    print(f"  Train CSV: {args.train_csv}")
    print(f"  Val CSV: {args.val_csv}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    
    # Load data
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Initialize DNABERT model
    config = DNABERTConfig(
        max_length=512,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs
    )
    
    dnabert = DNABERTModel(config=config)
    tokenizer = dnabert.load_tokenizer()
    model = dnabert.load_model()
    
    print("Model loaded successfully")
    
    # Create datasets
    train_dataset = PromoterDNABERTDataset(
        sequences=train_df['sequence'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    
    val_dataset = PromoterDNABERTDataset(
        sequences=val_df['sequence'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    
    # Create dataloaders
    train_loader = train_dataset.create_dataloader(batch_size=config.batch_size, shuffle=True)
    val_loader = val_dataset.create_dataloader(batch_size=config.batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    trainer = DNABERTTrainer(
        model=model,
        device=device,
        config=config.__dict__
    )
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.output_dir
    )
    
    # Save final model
    dnabert.save(args.output_dir)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(args.output_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {args.output_dir}")
    print(f"Training history saved to: {history_path}")

if __name__ == "__main__":
    main()
