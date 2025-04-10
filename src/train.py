import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import logging
from typing import Dict, Any, List, Tuple
import numpy as np

from models.joint_model import JointCTCAttentionModel
from data.dataset import CuedSpeechDataset, create_dataloader
from utils.text_processing import load_vocabulary, remove_blank_tokens
from utils.metrics import evaluate_model, calculate_per, calculate_cer, calculate_gesture_accuracy

def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, 
                   epoch: int, loss: float, save_path: str) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)

def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, 
                   checkpoint_path: str) -> int:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def decode_ctc_output(ctc_logits: torch.Tensor, blank_idx: int = 0) -> List[List[int]]:
    arg_maxes = torch.argmax(ctc_logits, dim=2)  # Get the most likely class for each time step
    decodes = []
    for args in arg_maxes:
        args = torch.unique_consecutive(args)  # Remove consecutive repeated indices
        decode = []
        for index in args:
            if index != blank_idx:
                decode.append(index.item())  # Append non-blank and non-repeated tokens
        decodes.append(decode)
    return decodes

def train_epoch(model: nn.Module, dataloader: DataLoader, vocab: Dict[str, int], 
                optimizer: optim.Optimizer, device: torch.device,
                loss_alpha: float = 0.5) -> float:
    model.train()
    total_loss = 0
    blank_idx = next((idx for token, idx in vocab.items() if token == '<UNK>'), None)
    
    for batch in tqdm(dataloader, desc='Training', disable=True):
        hand_shape = batch[0].to(device)
        hand_pos = batch[1].to(device)
        lips = batch[2].to(device)
        targets = batch[3].to(device)
        
        label_lengths = (targets != blank_idx).sum(dim=1).to(device)
        ctc_logits, attn_logits = model(hand_shape, hand_pos, lips, targets)
        input_lengths = torch.full((hand_shape.size(0),), ctc_logits.size(1), dtype=torch.long, device=device)
       
        loss = model.compute_loss(
            ctc_logits=ctc_logits, 
            attn_logits=attn_logits, 
            target_seq=targets,
            input_lengths=input_lengths, 
            label_lengths=label_lengths,
            alpha=loss_alpha
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model: nn.Module, dataloader: DataLoader, 
            device: torch.device, mode: str, vocab: Dict[str, int]) -> Dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    blank_idx = next((idx for token, idx in vocab.items() if token == '<UNK>'), None)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', disable=True):
            hand_shape = batch[0].to(device)
            hand_pos = batch[1].to(device)
            lips = batch[2].to(device)
            batch_targets = batch[3].to(device)
            
            ctc_logits, _ = model(hand_shape, hand_pos, lips, target_seq=None)
            
            decoded_seqs = decode_ctc_output(ctc_logits, blank_idx)
            
            predictions.extend(decoded_seqs)
            targets.extend(batch_targets.cpu().numpy().tolist())
    idx_to_token = {idx: token for token, idx in vocab.items()}
    pred_tokens = [[idx_to_token[idx] for idx in seq if idx != 1] for seq in predictions]
    target_tokens = [[idx_to_token[idx] for idx in seq if idx != 1] for seq in targets]
    
    metrics = {}
    
    if mode == 'phoneme':
        per = calculate_per(pred_tokens, target_tokens)
        metrics['per'] = per
        metrics['accuracy'] = 1 - per
    
    elif mode == 'syllable':
        cer = calculate_cer(pred_tokens, target_tokens)
        ser = calculate_per(pred_tokens, target_tokens)
        print("True syllables: ", target_tokens[:5])
        print("decoded syllables: ", pred_tokens[:5])
        ger = calculate_gesture_accuracy(pred_tokens, target_tokens)
        
        metrics['cer'] = cer
        metrics['ser'] = ser
        metrics['ger'] = ger
        
        
        sys.stdout.flush()
    return metrics

def main(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_logging(args.log_dir)
    logging.info(f"Using device: {device}")
    logging.info(f"Training mode: {args.mode}")
    
    logging.info(f"Loading vocabulary from: {args.vocab_path}")
    vocab = load_vocabulary(args.vocab_path)
    vocab_size = len(vocab)
    logging.info(f"Vocabulary size: {vocab_size}")
    
    logging.info("Creating datasets and dataloaders...")
    try:
        train_dataset = CuedSpeechDataset(
            features_dir=args.train_features_dir,
            labels_dir=args.train_labels_dir,
            mode=args.mode,
            vocab=vocab
        )
        val_dataset = CuedSpeechDataset(
            features_dir=args.val_features_dir,
            labels_dir=args.val_labels_dir,
            mode=args.mode,
            vocab=vocab
        )
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}. Please check feature/label directories.")
        return
        
    train_loader = create_dataloader(
        features_dir=args.train_features_dir,
        labels_dir=args.train_labels_dir,
        mode=args.mode,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        vocab=vocab
    )
    val_loader = create_dataloader(
        features_dir=args.val_features_dir,
        labels_dir=args.val_labels_dir,
        mode=args.mode,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        vocab=vocab
    )
    logging.info(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
    logging.info(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}")

    logging.info("Initializing model...")
    sample_hand_shape, sample_hand_pos, sample_lips, _ = train_dataset[0]
    hand_shape_dim = sample_hand_shape.shape[-1]
    hand_pos_dim = sample_hand_pos.shape[-1]
    lip_dim = sample_lips.shape[-1]
    logging.info(f"Inferred Feature Dims: Hand Shape={hand_shape_dim}, Hand Pos={hand_pos_dim}, Lips={lip_dim}")

    model = JointCTCAttentionModel(
        hand_shape_dim=hand_shape_dim,
        hand_pos_dim=hand_pos_dim,
        lips_dim=lip_dim,
        output_dim=vocab_size,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    start_epoch = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        logging.info(f"Loading checkpoint from: {args.checkpoint_path}")
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint_path)
        logging.info(f"Resuming training from epoch {start_epoch + 1}")
    elif args.checkpoint_path:
         logging.warning(f"Checkpoint path provided but not found: {args.checkpoint_path}")

    best_val_metric = float('inf')
    logging.info(f"Starting training for {args.num_epochs - start_epoch} epochs...")
    for epoch in range(start_epoch, args.num_epochs):
        train_loss = train_epoch(model, train_loader, vocab, optimizer, device, args.mode, args.loss_alpha)
        logging.info(f'Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f}')
        
        val_metrics = validate(model, val_loader, device, args.mode, vocab)
        
        if args.mode == 'phoneme':
            log_msg = f'Validation - PER: {val_metrics["per"]:.4f}, Accuracy: {val_metrics["accuracy"]:.4f}'
            current_metric = val_metrics['per']
            metric_name = "PER"
        else:
            log_msg = f'Validation - CER: {val_metrics["cer"]:.4f}, SER: {val_metrics["ser"]:.4f}, GER: {val_metrics["ger"]:.4f}'
            current_metric = val_metrics['ser']
            metric_name = "SER"
        logging.info(log_msg)
        
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        if current_metric < best_val_metric:
            best_val_metric = current_metric
            best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch+1, train_loss, best_model_path)
            logging.info(f"Saved best model (Epoch {epoch+1}, {metric_name}: {best_val_metric:.4f}) to {best_model_path}")
        print()
    logging.info("Loading best model for final validation metric calculation...")
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        _ = load_checkpoint(model, optimizer, best_model_path)
        final_val_metrics = validate(model, val_loader, device, args.mode, vocab)
        logging.info("--- Final Validation Metrics (Best Model) ---")
        if args.mode == 'phoneme':
             logging.info(f"PER: {final_val_metrics['per']:.4f}, Accuracy: {final_val_metrics['accuracy']:.4f}")
        else:
             logging.info(f"CER: {final_val_metrics['cer']:.4f}, SER: {final_val_metrics['ser']:.4f}, GER: {final_val_metrics['ger']:.4f}")
    else:
        logging.warning("Best model checkpoint not found. Skipping final metric calculation.")

    logging.info("Training completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Cued Speech Recognition Model')
    
    parser.add_argument('--train_features_dir', type=str, default='data/training_features', help='Directory for training features')
    parser.add_argument('--train_labels_dir', type=str, default='data/train_labels', help='Directory for training labels')
    parser.add_argument('--val_features_dir', type=str, default='data/val_features', help='Directory for validation features')
    parser.add_argument('--val_labels_dir', type=str, default='data/val_labels', help='Directory for validation labels')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.txt', help='Path to vocabulary file')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size for encoder/decoder')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers for GRUs')
    parser.add_argument('--mode', type=str, choices=['phoneme', 'syllable'], default="syllable", help='Training mode: phoneme or syllable level')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=3000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for Adam optimizer')
    parser.add_argument('--loss_alpha', type=float, default=0.2, help='Weight for CTC loss in joint loss (1-alpha for Attention loss)')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers for DataLoader')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_model.pt', help='Path to load a checkpoint to resume training')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.vocab_path):
         print(f"Error: Vocab path not found: {args.vocab_path}")
         exit()
    if not os.path.isdir(args.train_features_dir):
         print(f"Error: Train features directory not found: {args.train_features_dir}")
         exit()

    main(args) 