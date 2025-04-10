import jiwer
from typing import List, Dict, Tuple
import numpy as np
from .text_processing import syllables_to_gestures, syllables_to_phonemes, gestures_to_chars

def calculate_per(predictions: List[List[str]], targets: List[List[str]]) -> float:
    """
    Calculate Phoneme Error Rate (PER).
    
    Args:
        predictions: List of predicted phoneme sequences
        targets: List of target phoneme sequences
        
    Returns:
        PER value
    """
    # Convert sequences to space-separated strings
    
    pred_str = [" ".join([char for char in seq ]) for seq in predictions]
    target_str = [" ".join([char for char in seq ]) for seq in targets]
    
    # Calculate PER using jiwer
    per = jiwer.wer(target_str, pred_str)
    return per

def calculate_cer(predictions: List[List[str]], targets: List[List[str]]) -> float:
    """
    Calculate Character Error Rate (CER).
    
    Args:
        predictions: List of predicted character sequences
        targets: List of target character sequences
        
    Returns:
        CER value
    """
    # Convert sequences to strings
    pred_phonemes = [" ".join([char for char in syllables_to_phonemes(seq)]) for seq in predictions]
    target_phonemes = [" ".join([char for char in syllables_to_phonemes(seq)]) for seq in targets]
    
    # Calculate CER using jiwer
    cer = jiwer.wer(target_phonemes, pred_phonemes)
    return cer

def calculate_gesture_accuracy(predictions: List[List[str]], targets: List[List[str]]) -> float:
    """
    Calculate gesture accuracy.
    
    Args:
        predictions: List of predicted gesture sequences
        targets: List of target gesture sequences
        
    Returns:
        Gesture accuracy (1 - GER)
    """
    # Convert gestures to single-character representations
    pred_chars = [gestures_to_chars(syllables_to_gestures(pred)) for pred in predictions]
    target_chars = [gestures_to_chars(syllables_to_gestures(target)) for target in targets]
    
    print("Target gestures: ", [syllables_to_gestures(target) for target in targets][:5])
    print("Decoded gestures: ", [syllables_to_gestures(pred) for pred in predictions][:5])

    # Calculate GER using jiwer
    pred_str = [' '.join(pred) for pred in pred_chars]
    target_str = [' '.join(target) for target in target_chars]
    print("gesture to chars: ", pred_str[:10])
    ger = jiwer.wer(target_str, pred_str)
    
    # Return accuracy (1 - error rate)
    return ger

def compute_edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """
    Compute Levenshtein distance between two sequences.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        Edit distance
    """
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m + 1, n + 1))
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    
    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = min(
                    dp[i-1, j] + 1,    # deletion
                    dp[i, j-1] + 1,    # insertion
                    dp[i-1, j-1] + 1   # substitution
                )
    
    return int(dp[m, n])

def evaluate_model(predictions: List[List[str]], targets: List[List[str]], 
                  mode: str = 'phoneme') -> Dict[str, float]:
    """
    Evaluate model predictions.
    
    Args:
        predictions: List of predicted sequences
        targets: List of target sequences
        mode: Evaluation mode ('phoneme', 'syllable', or 'gesture')
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    if mode == 'phoneme':
        metrics['per'] = calculate_per(predictions, targets)
    
    elif mode == 'syllable':
        # Calculate syllable error rate
        metrics['ser'] = calculate_per(predictions, targets)
        
        # Convert syllables to phonemes for CER
        pred_phonemes = [syllables_to_phonemes(pred) for pred in predictions]
        target_phonemes = [syllables_to_phonemes(target) for target in targets]
        metrics['cer'] = calculate_per(pred_phonemes, target_phonemes)
        
        # Calculate gesture accuracy and error rate
        gesture_acc = calculate_gesture_accuracy(predictions, targets)
        metrics['ger'] = 1.0 - gesture_acc
    
    elif mode == 'gesture':
        # Convert syllables to gestures
        pred_gestures = [syllables_to_gestures(pred) for pred in predictions]
        target_gestures = [syllables_to_gestures(target) for target in targets]
        metrics['gesture_accuracy'] = calculate_gesture_accuracy(pred_gestures, target_gestures)
    
    return metrics 