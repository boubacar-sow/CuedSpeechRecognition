import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
import numpy as np
import kenlm
from collections import defaultdict, Counter
from torch.utils.data import DataLoader
import jiwer

from models.joint_model import JointCTCAttentionModel
from data.dataset import CuedSpeechDataset, create_dataloader
from utils.text_processing import load_vocabulary, remove_blank_tokens, syllables_to_gestures, syllables_to_phonemes, gestures_to_chars
from utils.metrics import calculate_per, calculate_cer, calculate_gesture_accuracy

def unique(sequence):
    """
    Keep only unique items in a sequence while preserving their original order.
    
    Args:
        sequence: Input sequence
        
    Returns:
        Sequence with duplicates removed while preserving order
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'testing.log')),
            logging.StreamHandler()
        ]
    )

def prefix_beam_search(ctc, index_to_phoneme, blank_idx, space_idx, eos_idx, sos_idx,
                      lm=None, k=25, alpha=0.30, beta=5, prune=0.00001):
    lm = (lambda s: 1) if lm is None else lm

    import torch.nn.functional as F
    ctc = F.softmax(ctc, dim=-1)
    F = ctc.shape[1]
    ctc = np.vstack((np.zeros(F), ctc.cpu()))
    T = ctc.shape[0]
    
    def get_lm_text(seq):
        tokens = [index_to_phoneme[i] for i in seq if i not in (blank_idx, sos_idx, eos_idx)]
        tokens = [(" " if i == index_to_phoneme[space_idx] else i) for i in tokens]
        text = "".join(tokens).strip()
        return text
    
    def count_words(seq):
        text = get_lm_text(seq)
        count = len(text.split()) if text else 0
        return count

    O = ()
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][O] = 1
    Pnb[0][O] = 0
    A_prev = [O]

    for t in range(1, T):
        pruned_indices = np.where(ctc[t] > prune)[0]

        for l in A_prev:
            if len(l) > 0 and l[-1] == eos_idx:
                Pb[t][l] = Pb[t - 1][l]
                Pnb[t][l] = Pnb[t - 1][l]
                continue

            for c in pruned_indices:
                if c == blank_idx:
                    Pb[t][l] += ctc[t][blank_idx] * (Pb[t - 1][l] + Pnb[t - 1][l])
                else:
                    l_plus = l + (c,)
                    if len(l) > 0 and c == l[-1]:
                        Pnb[t][l_plus] += ctc[t][c] * Pb[t - 1][l]
                        Pnb[t][l] += ctc[t][c] * Pnb[t - 1][l]
                    elif len([x for x in l if x != space_idx]) > 0 and c in (space_idx, eos_idx):
                        lm_prob = lm(get_lm_text(l_plus)) ** alpha
                        Pnb[t][l_plus] += lm_prob * ctc[t][c] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    else:
                        Pnb[t][l_plus] += ctc[t][c] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    
                    if l_plus not in A_prev:
                        Pb[t][l_plus] += ctc[t][blank_idx] * (Pb[t - 1][l_plus] + Pnb[t - 1][l_plus])
                        Pnb[t][l_plus] += ctc[t][c] * Pnb[t - 1][l_plus]
        
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] * ((count_words(l) + 1) ** beta)
        A_prev = sorted(A_next, key=sorter, reverse=True)[:k]

    best = A_prev[0]
    return list(best)

def decode_ctc_output(ctc_logit: torch.Tensor, blank_idx: int = 0) -> List[List[int]]:
    arg_maxes = torch.argmax(ctc_logit, dim=-1)  # Get the most likely class for each time step
    decodes = []
    args = torch.unique_consecutive(arg_maxes)  # Remove consecutive repeated indices
    for index in args:
        if index != blank_idx:
            decodes.append(index.item())  # Append non-blank and non-repeated tokens
    return decodes

def evaluate_model(model: nn.Module, dataloader: DataLoader, 
                  device: torch.device, mode: str, vocab: Dict[str, int],
                  use_beam_search: bool = False, beam_size: int = 5, 
                  lm_path: str = None, alpha: float = 0.30, beta: float = 5) -> Dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    
    if use_beam_search and lm_path:
        try:
            kenlm_model = kenlm.Model(lm_path)
            
            def lm_score(text):
                tokens = text.split()
                if not tokens:
                    return 1.0
                    
                state_in = kenlm.State()
                state_out = kenlm.State()
                kenlm_model.BeginSentenceWrite(state_in)
                
                for token in tokens[:-1]:
                    _ = kenlm_model.BaseScore(state_in, token, state_out)
                    state_in, state_out = state_out, state_in
                    
                last_token = tokens[-1] if len(tokens) >= 1 else "dfjknd"
                last_token_logprob = kenlm_model.BaseScore(state_in, last_token, state_out)
                return 10 ** last_token_logprob
        except Exception as e:
            logging.warning(f"Failed to load language model: {e}. Using dummy LM.")
            lm_score = lambda s: 1.0
    else:
        lm_score = lambda s: 1.0
    
    blank_idx = next((idx for token, idx in vocab.items() if token == '<UNK>'), None)
    space_idx = next((idx for token, idx in vocab.items() if token == '_'), None)
    eos_idx = next((idx for token, idx in vocab.items() if token == '<EOS>'), None)
    sos_idx = next((idx for token, idx in vocab.items() if token == '<SOS>'), None)
    
    idx_to_token = {idx: token for token, idx in vocab.items()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            hand_shape = batch[0].to(device)
            hand_pos = batch[1].to(device)
            lips = batch[2].to(device)
            batch_targets = batch[3].to(device)
            
            ctc_logits, _ = model(hand_shape, hand_pos, lips, target_seq=None)
            
            for i in range(ctc_logits.size(0)):
                sample_logits = ctc_logits[i]
                
                if use_beam_search:
                    decoded_indices = prefix_beam_search(
                        sample_logits, 
                        idx_to_token,
                        blank_idx=blank_idx,
                        space_idx=space_idx,
                        eos_idx=eos_idx,
                        sos_idx=sos_idx,
                        lm=lm_score,
                        k=beam_size,
                        alpha=alpha,
                        beta=beta
                    )
                else:
                    decoded_indices = decode_ctc_output(sample_logits, blank_idx)
                
                valid_indices = [idx for idx in decoded_indices if idx < len(vocab)]
                decoded_tokens = [idx_to_token.get(idx, '<UNK>') for idx in valid_indices]
                predictions.append(decoded_tokens)
            
            for sequence in batch_targets:
                valid_indices = [idx.item() for idx in sequence if idx != blank_idx and idx.item() < len(vocab)]
                true_tokens = [idx_to_token.get(idx, '<UNK>') for idx in valid_indices 
                               if idx_to_token.get(idx, '<UNK>') != '_']
                targets.append(true_tokens)
    
    metrics = {}
    
    if mode == 'phoneme':
        per = calculate_per(predictions, targets)
        metrics['cer'] = per
    
    elif mode == 'syllable':
        # Calculate syllable error rate
        pred_tokens = predictions
        target_tokens = targets
        cer = calculate_cer(pred_tokens, target_tokens)
        ser = calculate_per(pred_tokens, target_tokens)
        print("True syllables: ", target_tokens[:5])
        print("decoded syllables: ", pred_tokens[:5])
        ger = calculate_gesture_accuracy(pred_tokens, target_tokens)
        
        metrics['cer'] = cer
        metrics['ser'] = ser
        metrics['ger'] = ger
    
    return metrics

def main(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_logging(args.log_dir)
    logging.info(f"Using device: {device}")
    logging.info(f"Testing mode: {args.mode}")

    logging.info(f"Loading vocabulary from: {args.vocab_path}")
    vocab = load_vocabulary(args.vocab_path)
    unique_vocab_keys = unique(list(vocab.keys()))
    vocab = {k: i for i, k in enumerate(unique_vocab_keys)}
    vocab_size = len(vocab)
    logging.info(f"Vocabulary size: {vocab_size}")
    idx_to_token = {idx: token for token, idx in vocab.items()}

    logging.info("Creating test dataset and dataloader...")
    try:
        test_dataset = CuedSpeechDataset(
            features_dir=args.test_features_dir,
            labels_dir=args.test_labels_dir,
            vocab=vocab,
            mode=args.mode
        )
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}. Please check feature/label directories.")
        return
    
    test_loader = create_dataloader(
        features_dir=args.test_features_dir,
        labels_dir=args.test_labels_dir,
        mode=args.mode,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        vocab=vocab
    )
    
    logging.info(f"Test dataset size: {len(test_dataset)}, Test loader batches: {len(test_loader)}")

    logging.info("Initializing model...")
    try:
        sample_hand_shape, sample_hand_pos, sample_lips, _ = test_dataset[0]
        hand_shape_dim = sample_hand_shape.shape[-1]
        hand_pos_dim = sample_hand_pos.shape[-1]
        lip_dim = sample_lips.shape[-1]
        logging.info(f"Inferred Feature Dims: Hand Shape={hand_shape_dim}, Hand Pos={hand_pos_dim}, Lips={lip_dim}")
    except IndexError:
        logging.error("Test dataset appears to be empty. Cannot infer feature dimensions.")
        return

    model = JointCTCAttentionModel(
        hand_shape_dim=hand_shape_dim,
        hand_pos_dim=hand_pos_dim,
        lips_dim=lip_dim,
        output_dim=vocab_size,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers
    ).to(device)
    
    if not args.checkpoint_path or not os.path.exists(args.checkpoint_path):
        logging.error(f"Checkpoint path not provided or not found: {args.checkpoint_path}")
        return
    logging.info(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    logging.info(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")

    logging.info("--- Evaluating with Regular (Greedy) CTC Decoding ---")
    regular_metrics = evaluate_model(
        model=model, 
        dataloader=test_loader, 
        device=device, 
        mode=args.mode, 
        vocab=vocab,
        use_beam_search=False
    )
    
    if args.mode == 'phoneme':
        logging.info(f"Regular Decoding - CER: {regular_metrics.get('cer', float('nan')):.4f}")
    else:
        logging.info(f"Regular Decoding - CER: {regular_metrics.get('cer', float('nan')):.4f}")
        logging.info(f"Regular Decoding - SER: {regular_metrics.get('ser', float('nan')):.4f}")
        logging.info(f"Regular Decoding - GER: {regular_metrics.get('ger', float('nan')):.4f}")
    
    logging.info("--- Evaluating with Prefix Beam Search CTC Decoding ---")
    beam_metrics = evaluate_model(
        model=model, 
        dataloader=test_loader, 
        device=device, 
        mode=args.mode, 
        vocab=vocab,
        use_beam_search=True, 
        beam_size=args.beam_size,
        lm_path=args.lm_path, 
        alpha=args.alpha, 
        beta=args.beta
    )
    
    if args.mode == 'phoneme':
        logging.info(f"Beam Search (k={args.beam_size}, alpha={args.alpha}, beta={args.beta}) - CER: {beam_metrics.get('cer', float('nan')):.4f}")
    else:
        logging.info(f"Beam Search (k={args.beam_size}, alpha={args.alpha}, beta={args.beta}) - CER: {beam_metrics.get('cer', float('nan')):.4f}")
        logging.info(f"Beam Search (k={args.beam_size}, alpha={args.alpha}, beta={args.beta}) - SER: {beam_metrics.get('ser', float('nan')):.4f}")
        logging.info(f"Beam Search (k={args.beam_size}, alpha={args.alpha}, beta={args.beta}) - GER: {beam_metrics.get('ger', float('nan')):.4f}")
    
    logging.info("--- Beam Search Improvement vs Regular Decoding ---")
    if args.mode == 'phoneme':
        improvement = regular_metrics.get('cer', 0) - beam_metrics.get('cer', 0)
        logging.info(f"CER Improvement: {improvement:.4f}")
    else:
        cer_improvement = regular_metrics.get('cer', 0) - beam_metrics.get('cer', 0)
        ser_improvement = regular_metrics.get('ser', 0) - beam_metrics.get('ser', 0)
        ger_improvement = regular_metrics.get('ger', 0) - beam_metrics.get('ger', 0)
        logging.info(f"  CER Improvement: {cer_improvement:.4f}")
        logging.info(f"  SER Improvement: {ser_improvement:.4f}")
        logging.info(f"  GER Improvement: {ger_improvement:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Cued Speech Recognition Model')
    
    parser.add_argument('--test_features_dir', type=str, default='../data/testing_features', help='Directory for test features')
    parser.add_argument('--test_labels_dir', type=str, default='../data/test_labels', help='Directory for test labels')
    parser.add_argument('--vocab_path', type=str, default='../data/vocab.txt', help='Path to vocabulary file')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size for encoder/decoder')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers for GRUs')
    parser.add_argument('--mode', type=str, choices=['phoneme', 'syllable'], default="syllable", help='Testing mode: phoneme or syllable level')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing')
    parser.add_argument('--beam_size', type=int, default=2, help='Beam width for prefix beam search')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for the LM score')
    parser.add_argument('--beta', type=float, default=1.0, help='Word bonus exponent')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--lm_path', type=str, default='../data/french_ipa.binary', help='Path to the KenLM language model binary file')
    parser.add_argument('--checkpoint_path', type=str, default='../checkpoints/best_model.pt', help='Path to the model checkpoint file (.pt)')
    parser.add_argument('--log_dir', type=str, default='../logs', help='Directory to save logs')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.vocab_path):
         print(f"Error: Vocab path not found: {args.vocab_path}")
         exit()
    if not os.path.isdir(args.test_features_dir):
         print(f"Error: Test features directory not found: {args.test_features_dir}")
         exit()
    if args.lm_path and not os.path.exists(args.lm_path):
         print(f"Warning: Language model path specified but not found: {args.lm_path}. Beam search will proceed without LM.")
         args.lm_path = None

    main(args) 