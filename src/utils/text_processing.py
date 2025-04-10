from typing import List, Dict, Set
import csv
import os

# Phoneme sets
CONSONANTS = {
    "b", "d", "f", "g", "h", "j", "k", "l", "m", "n", "n~", "p", "r", "s", 
    "s^", "t", "v", "w", "z", "z^", "ng", "gn"
}

VOWELS = {
    "a", "a~", "e", "e^", "e~", "i", "o", "o^", "o~", "u", "y", "x", "x^", "x~"
}

# Mappings for gesture conversion
consonant_to_handshapes = {
    "b": 1, "p": 1, "m": 1,
    "d": 2, "t": 2, "n": 2, "n~": 2, "ng": 2,
    "g": 3, "k": 3, "gn": 3,
    "v": 4, "f": 4,
    "z": 5, "s": 5, "z^": 5, "s^": 5,
    "j": 6, "h": 6,
    "r": 7, "l": 7,
    "w": 8
}

vowel_to_position = {
    "i": 1, "e": 1, "e^": 1,
    "y": 2, "x": 2, "x^": 2,
    "u": 3, "o": 3, "o^": 3,
    "a": 4, "a~": 4,
    "e~": 5, "x~": 5, "o~": 5
}

# Mapping of gestures to single characters for GER calculation
# This maps each possible handshape-position combination and special tokens
# to a unique character
GESTURE_TO_CHAR = {
    # Handshape 1 (positions 1-5)
    "1-1": "a", "1-2": "b", "1-3": "c", "1-4": "d", "1-5": "e",
    
    # Handshape 2 (positions 1-5)
    "2-1": "f", "2-2": "g", "2-3": "h", "2-4": "i", "2-5": "j",
    
    # Handshape 3 (positions 1-5)
    "3-1": "k", "3-2": "l", "3-3": "m", "3-4": "n", "3-5": "o",
    
    # Handshape 4 (positions 1-5)
    "4-1": "p", "4-2": "q", "4-3": "r", "4-4": "s", "4-5": "t",
    
    # Handshape 5 (positions 1-5)
    "5-1": "u", "5-2": "v", "5-3": "w", "5-4": "x", "5-5": "y",
    
    # Handshape 6 (positions 1-5)
    "6-1": "z", "6-2": "A", "6-3": "B", "6-4": "C", "6-5": "D",
    
    # Handshape 7 (positions 1-5)
    "7-1": "E", "7-2": "F", "7-3": "G", "7-4": "H", "7-5": "I",
    
    # Handshape 8 (positions 1-5)
    "8-1": "J", "8-2": "K", "8-3": "L", "8-4": "M", "8-5": "N",
    
    # Special tokens
    "<SOS>": "O", "<EOS>": "P", "<PAD>": "Q", "<UNK>": "R", "_": "S"
}

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

def load_vocabulary(vocab_path: str) -> Dict[str, int]:
    """
    Load vocabulary from file.
    
    Args:
        vocab_path: Path to vocabulary file
        
    Returns:
        Dictionary mapping phonemes to indices
    """
    with open(vocab_path, 'r') as f:
        reader = csv.reader(f)
        vocabulary = [row[0] for row in reader]
    
    # Ensure vocabulary contains only unique items while preserving order
    vocabulary = unique(vocabulary)
    
    # Create mappings
    phoneme_to_index = {phoneme: idx for idx, phoneme in enumerate(vocabulary)}
    return phoneme_to_index

def syllabify_ipa(ipa_text: str) -> List[str]:
    """
    Convert IPA text to syllables.
    
    Args:
        ipa_text: Space-separated IPA phonemes
        
    Returns:
        List of syllables
    """
    phonemes = ipa_text.split()
    syllables = []
    i = 0
    
    while i < len(phonemes):
        phone = phonemes[i]
        
        # Handle special tokens
        if phone == '_':
            i += 1
            syllables.append(phone)
            continue
        
        # Process vowels
        if phone in VOWELS:
            syllables.append(phone)
            i += 1
        
        # Process consonants
        elif phone in CONSONANTS:
            if i + 1 < len(phonemes):
                next_phone = phonemes[i + 1]
                
                # Handle special cases
                if next_phone == '_':
                    i += 1
                    syllables.append(phone)
                    continue
                
                if next_phone in VOWELS:
                    # CV syllable
                    syllable = phone + next_phone
                    syllables.append(syllable)
                    i += 2
                else:
                    if next_phone == "q":
                        syllables.append(phone)
                        i += 2
                    else:
                        syllables.append(phone)
                        i += 1
            else:
                syllables.append(phone)
                i += 1
        
        else:
            i += 1
    
    return syllables

def syllables_to_gestures(syllable_sequence: List[str]) -> List[str]:
    """
    Convert a sequence of syllables into a sequence of gestures.
    
    Args:
        syllable_sequence: A list of syllables (strings).
        
    Returns:
        list: A list of gesture strings in the format "handshape-position".
    """
    gestures = []
    for syllable in syllable_sequence:
        # Check if the input is already a gesture string (e.g., "1-1" or "5-3")
        if '-' in syllable and len(syllable) == 3 and syllable[0].isdigit() and syllable[2].isdigit():
            gestures.append(syllable)
            continue
            
        # Handle special tokens
        if syllable in ["<SOS>", "<EOS>", "<PAD>", "<UNK>", "_"]:
            gestures.append(syllable)
        # Check if the syllable starts with a multi-character consonant (e.g., "s^")
        elif len(syllable) >= 3 and syllable[:2] in consonant_to_handshapes:
            consonant = syllable[:2]
            vowel = syllable[2:]  # Remaining part is the vowel
            handshape = consonant_to_handshapes.get(consonant, 5)  # Default handshape is 5
            position = vowel_to_position.get(vowel, 1)  # Default position is 1
            gestures.append(f"{handshape}-{position}")
        # Check if the syllable ends with a multi-character vowel (e.g., "me^")
        elif len(syllable) >= 3 and syllable[-2:] in vowel_to_position:
            consonant = syllable[:-2]  # Remaining part is the consonant
            vowel = syllable[-2:]
            handshape = consonant_to_handshapes.get(consonant, 5)  # Default handshape is 5
            position = vowel_to_position.get(vowel, 1)  # Default position is 1
            gestures.append(f"{handshape}-{position}")
        # Handle normal CV syllables (e.g., "ma")
        elif len(syllable) == 2:
            if syllable in consonant_to_handshapes:  # length 2 consonant only syllable
                handshape = consonant_to_handshapes.get(syllable, 5)  # Default handshape is 5
                position = 1  # Default position is 1
                gestures.append(f"{handshape}-{position}")
            elif syllable in vowel_to_position:  # length 2 vowel only syllable
                handshape = 5  # Default handshape is 5
                position = vowel_to_position.get(syllable, 1)
                gestures.append(f"{handshape}-{position}")
            elif syllable[0] in consonant_to_handshapes:  # Consonant-Vowel pair
                consonant = syllable[0]
                vowel = syllable[1]
                handshape = consonant_to_handshapes.get(consonant, 5)  # Default handshape is 5
                position = vowel_to_position.get(vowel, 1)  # Default position is 1
                gestures.append(f"{handshape}-{position}")
            elif syllable[0] in vowel_to_position:  # Vowel-only syllable
                vowel = syllable
                position = vowel_to_position.get(vowel, 1)  # Default position is 1
                gestures.append(f"5-{position}")  # Default handshape is 5
        # Handle C-only syllables (e.g., "m")
        elif len(syllable) == 1 and syllable in consonant_to_handshapes:
            handshape = consonant_to_handshapes.get(syllable, 5)  # Default handshape is 5
            gestures.append(f"{handshape}-1")  # Default position is 1
        # Handle V-only syllables (e.g., "a")
        elif len(syllable) == 1 and syllable in vowel_to_position:
            position = vowel_to_position.get(syllable, 1)  # Default position is 1
            gestures.append(f"5-{position}")  # Default handshape is 5
        else:
            # Unknown syllable
            print(f"Unknown syllable: {syllable}")
            gestures.append("<UNK>")
    return gestures

def gestures_to_chars(gesture_sequence: List[str]) -> str:
    """
    Convert a sequence of gestures to a string of single characters for GER calculation.
    
    Args:
        gesture_sequence: A list of gestures in the format "handshape-position"
        
    Returns:
        A string where each character represents one gesture
    """
    chars = []
    for gesture in gesture_sequence:
        if gesture in GESTURE_TO_CHAR:
            chars.append(GESTURE_TO_CHAR[gesture])
        else:
            # If the gesture doesn't have a mapping, use the unknown token character
            chars.append(GESTURE_TO_CHAR["<UNK>"])
    
    return ''.join(chars)

def syllables_to_phonemes(syllable_sequence: List[str]) -> List[str]:
    """
    Convert syllables to individual phonemes.
    
    Args:
        syllable_sequence: List of syllables
        
    Returns:
        List of phonemes
    """
    phonemes = []
    for syllable in syllable_sequence:
        if syllable in ["<SOS>", "<EOS>", "<PAD>", "<UNK>", "_", " "]:
            phonemes.append(syllable)
            continue
        
        # Handle multi-character consonants (e.g., "s^")
        if len(syllable) >= 3 and syllable[:2] in consonant_to_handshapes:
            consonant = syllable[:2]
            vowel = syllable[2:]  # Remaining part is the vowel
            phonemes.append(consonant)
            phonemes.append(vowel)
        
        # Handle multi-character vowels (e.g., "me^")
        elif len(syllable) >= 3 and syllable[-2:] in vowel_to_position:
            consonant = syllable[:-2]  # Remaining part is the consonant
            vowel = syllable[-2:]
            phonemes.append(consonant)
            phonemes.append(vowel)
        
        # Handle normal CV syllables (e.g., "ma")
        elif len(syllable) == 2:
            consonant = syllable[0]
            vowel = syllable[1]
            phonemes.append(consonant)
            phonemes.append(vowel)
        
        # Handle C-only syllables (e.g., "m")
        elif len(syllable) == 1 and syllable in consonant_to_handshapes:
            phonemes.append(syllable)
        
        # Handle V-only syllables (e.g., "a")
        elif len(syllable) == 1 and syllable in vowel_to_position:
            phonemes.append(syllable)
        
        else:
            # Unknown syllable
            print(f"Unknown syllable: {syllable}")
            phonemes.append("<UNK>")
    
    return phonemes

def compute_sequence_lengths(sequences: List[List[int]]) -> List[int]:
    """
    Compute sequence lengths for CTC loss.
    
    Args:
        sequences: List of sequences
        
    Returns:
        List of sequence lengths
    """
    return [len(seq) for seq in sequences]

def remove_blank_tokens(sequence: List[int], blank_idx: int = 0) -> List[int]:
    """
    Remove blank tokens from a sequence.
    
    Args:
        sequence: Input sequence
        blank_idx: Index of blank token
        
    Returns:
        Sequence without blank tokens
    """
    return [idx for idx in sequence if idx != blank_idx] 