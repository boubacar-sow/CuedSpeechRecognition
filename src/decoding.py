import csv
import os
from collections import defaultdict
import random
import numpy as np
import pickle

import jiwer
import numpy as np
import cv2
import time
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from praatio import textgrid as tgio
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import itertools
import torch
import torch.nn.functional as F
import sys
import gc
import sys
import os

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

#from acsr.utils.text_processing import syllabify_ipa  # Now works

import wandb  # Import W&B
from acsr.next_syllable_prediction import NextSyllableLSTM
from acsr.variables import consonant_to_handshapes, vowel_to_position


# Load CSV files from a directory based on a filename pattern
def load_csv_files(directory, filename_pattern):
    files_data = {}
    for filename in os.listdir(directory):
        if filename_pattern in filename:
            df = pd.read_csv(os.path.join(directory, filename))
            df.dropna(inplace=True)
            base_name = filename.split(filename_pattern)[0]
            if "sent_" in base_name:
                continue
            files_data[base_name] = df
    return files_data

# Find corresponding phoneme files based on the base names of position filenames
def find_phoneme_files(directory, base_names):
    phoneme_files = {}
    for base_name in base_names:
        phoneme_file = os.path.join(directory, f'{base_name}.csv')
        if os.path.exists(phoneme_file):
            phoneme_files[base_name] = phoneme_file
    return phoneme_files

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

# Load phoneme-to-index mapping
# with open(r"/pasteur/appa/homes/bsow/ACSR/data/training_videos/CSF22_train/phonelist.csv", "r") as file:
with open(r"/pasteur/appa/homes/bsow/ACSR/data/french_dataset/vocab.txt", "r") as file: 
    reader = csv.reader(file)
    vocabulary_list = [row[0] for row in reader]

phoneme_to_index = {phoneme: idx for idx, phoneme in enumerate(unique(vocabulary_list))}
index_to_phoneme = {idx: phoneme for phoneme, idx in phoneme_to_index.items()}

def load_features(directory, base_name):
    file_path = os.path.join(directory, f"{base_name}_features.csv")
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)  # Drop rows with NaN values
    return df

def syllabify_ipa(ipa_text):
    consonants = {"b", "d", "f", "g", "h", "j", "k", "l", "m", "n", "n~", "p", "r", "s", "s^", "t", "v", "w", "z", "z^", "ng", "gn"}
    vowels = {"a", "a~", "e", "e^", "e~", "i", "o", "o^", "o~", "u", "y", "x", "x^", "x~"}
    phonemes = ipa_text.split()
    syllables = []
    i = 0

    while i < len(phonemes):
        phone = phonemes[i]
        
        if phone == '_':
            i += 1
            syllables.append(phone)
            continue
        
        if phone in vowels:
            syllables.append(phone)
            i += 1
        elif phone in consonants:
            # Check if there is a next phone
            if i + 1 < len(phonemes):
                next_phone = phonemes[i + 1]
                if next_phone == '_':
                    i += 1
                    syllables.append(phone)
                    continue
                if next_phone in vowels:
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


from collections import Counter


def prepare_data_for_videos_no_sliding_windows(base_names, phoneme_files, features_dir, labels_dir, 
                                              phoneme_to_index, img_size=(28, 28),
                                              lips_dir="/pasteur/appa/homes/bsow/ACSR/data/training_videos/CSF22_train/lip_rois_mp4"):
    train_videos_data = {}
    syllable_counter = Counter()
    
    for base_name in sorted(base_names):
        if base_name in phoneme_files:
            # Load pre-extracted features
            features_df = load_features(features_dir, base_name)
            if 'frame_number' not in features_df.columns:
                raise ValueError(f"Feature file for {base_name} does not contain 'frame_number' column.")


            # Separate features into different modalities
            hand_shape_columns = [col for col in features_df.columns if "hand" in col and "face" not in col]
            hand_pos_columns = [col for col in features_df.columns if "face" in col]
            lip_columns = [col for col in features_df.columns if "lip" in col]

            # Extract features
            X_acoustic_hand_shape = features_df[hand_shape_columns].to_numpy()
            X_acoustic_hand_pos = features_df[hand_pos_columns].to_numpy()
            X_acoustic_lips = features_df[lip_columns].to_numpy()

            # Load and process phoneme labels
            labels_path = os.path.join(labels_dir, f"{base_name}.csv")
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Phoneme label file not found: {labels_path}")
            
            phoneme_labels = pd.read_csv(labels_path, header=None).squeeze().tolist()[1:-1]
            syllable_labels = ["<SOS>"] + syllabify_ipa(" ".join(phoneme_labels)) + ["<EOS>"]
            
            # Convert to indices
            syllable_indices = []
            for syllable in syllable_labels:
                if syllable not in phoneme_to_index:
                    raise ValueError(f"Syllable '{syllable}' not found in vocabulary. File: {base_name}")
                syllable_indices.append(phoneme_to_index[syllable])
                syllable_counter[syllable] += 1

            # Store all data modalities
            train_videos_data[base_name] = {
                "X_acoustic_hand_shape": X_acoustic_hand_shape,
                "X_acoustic_hand_pos": X_acoustic_hand_pos,
                "X_acoustic_lips": X_acoustic_lips,
                "X_visual_lips": None,  # New visual modality
                "y": syllable_indices,
            }
    
    return train_videos_data, syllable_counter


import torch
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    hand_shape, hand_pos, acoustic_lips, labels = zip(*batch)
    
    # Pad sequences for hand shape, hand position, and acoustic lips
    hand_shape_padded = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in hand_shape], batch_first=True, padding_value=0)
    hand_pos_padded = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in hand_pos], batch_first=True, padding_value=0)
    acoustic_lips_padded = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in acoustic_lips], batch_first=True, padding_value=0)
    
    # Pad visual lips (image sequences)
    #visual_lips_padded = pad_sequence(
    #    [torch.tensor(x, dtype=torch.float32) for x in visual_lips],
    #    batch_first=True,
    #    padding_value=0  # Use zero-padding for images
    #)
    
    # Pad labels
    labels_padded = pad_sequence([torch.tensor(y, dtype=torch.long) for y in labels], batch_first=True, padding_value=phoneme_to_index["<UNK>"])
    
    return hand_shape_padded, hand_pos_padded, acoustic_lips_padded, labels_padded

from torch.utils.data import DataLoader

def data_to_dataloader(data, batch_size=4, shuffle=True):
    dataset = list(zip(
        data['X_acoustic_hand_shape'],
        data['X_acoustic_hand_pos'],
        data['X_acoustic_lips'],
        #data['X_visual_lips'],  # Include visual lips
        data['y']
    ))
    
    # Create a DataLoader with the custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=6,
        pin_memory=True,
        collate_fn=custom_collate_fn  # Use the custom collate function
    )
    return dataloader


# acoustic model
# acoustic model
import torch.nn as nn
from conformer import ConformerBlock

class ThreeStreamFusionEncoder(nn.Module):
    def __init__(self, hand_shape_dim, hand_pos_dim, lips_dim, visual_lips_dim, hidden_dim=128, n_layers=2):
        super(ThreeStreamFusionEncoder, self).__init__()

        self.hand_shape_gru = nn.GRU(hand_shape_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        self.hand_pos_gru   = nn.GRU(hand_pos_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        self.lips_gru       = nn.GRU(lips_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        
        # Fusion GRU: note the input size is 3 streams * 2 (bidirectional)
        self.fusion_gru = nn.GRU(hidden_dim * 6, hidden_dim * 3, n_layers, bidirectional=True, batch_first=True)
        
        # CNN for visual lips (unchanged)
        self.visual_lips_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, hidden_dim)),  # (batch, 64, 1, 64)
            nn.Flatten(start_dim=1),                # (batch, 4096)
            nn.Linear(128 * hidden_dim, hidden_dim)   # nn.Linear(4096, 64) if hidden_dim=64
        )

        # Cross-modal attention and fusion conformer remain the same
        self.cross_modal_attention = nn.MultiheadAttention(embed_dim=hidden_dim*6, num_heads=4)
        #self.fusion_conformer = nn.Sequential(*[ConformerBlock(dim=hidden_dim * 3) for _ in range(n_layers)])
    
    def forward(self, hand_shape, hand_pos, lips, visual_lips):
        # Project and process each modality with its respective Conformer
        hand_shape_out, _ = self.hand_shape_gru(hand_shape)  # (batch, seq, hidden_dim*2)
        hand_pos_out, _   = self.hand_pos_gru(hand_pos)
        lips_out, _       = self.lips_gru(lips)

        # Concatenate all modalities along the last dimension
        combined_features = torch.cat([hand_shape_out, hand_pos_out, lips_out], dim=-1)
        
        # Apply cross-modal attention
        #attn_output, _ = self.cross_modal_attention(combined_features, combined_features, combined_features)
        
        # Fusion Conformer
        fusion_out, _ = self.fusion_gru(combined_features)
        return fusion_out


class AttentionDecoder(nn.Module):
    def __init__(self, encoder_dim, output_dim, hidden_dim_decoder=None, n_layers=1):
        super(AttentionDecoder, self).__init__()
        # If not provided, set hidden_dim_decoder equal to encoder_dim
        if hidden_dim_decoder is None:
            hidden_dim_decoder = encoder_dim
        self.embedding = nn.Embedding(output_dim, hidden_dim_decoder)
        self.gru = nn.GRU(hidden_dim_decoder + encoder_dim, hidden_dim_decoder, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim_decoder, output_dim)
        self.encoder_dim = encoder_dim
        self.hidden_dim_decoder = hidden_dim_decoder

        # If needed, add a projection if hidden_dim_decoder != encoder_dim.
        # self.proj = nn.Linear(hidden_dim_decoder, encoder_dim)

    def forward(self, encoder_outputs, target_seq):
        """
        Args:
            encoder_outputs: Tensor of shape (batch, T, encoder_dim)
            target_seq: Tensor of shape (batch, target_len) containing target indices
        Returns:
            outputs: Tensor of shape (batch, target_len, output_dim)
        """
        batch_size, target_len = target_seq.size()
        hidden = None  # Alternatively, initialize hidden state here.
        outputs = []

        # For each time step in the target sequence (using teacher forcing)
        for t in range(target_len):
            # Get embedding for current target token: shape (batch, 1, hidden_dim_decoder)
            embedded = self.embedding(target_seq[:, t].long()).unsqueeze(1)
            
            # Dot-product attention:
            # Compute attention scores by dot-product between embedded and all encoder outputs.
            # embedded: (batch, 1, hidden_dim_decoder)
            # If hidden_dim_decoder != encoder_dim, you might project embedded via self.proj first.
            attn_scores = torch.bmm(embedded, encoder_outputs.transpose(1, 2))  # shape: (batch, 1, T)
            attn_weights = F.softmax(attn_scores, dim=-1)  # shape: (batch, 1, T)
            
            # Compute context vector as weighted sum of encoder outputs: shape (batch, 1, encoder_dim)
            attn_applied = torch.bmm(attn_weights, encoder_outputs)
            
            # Concatenate embedded input and context vector
            gru_input = torch.cat([embedded, attn_applied], dim=2)  # shape: (batch, 1, hidden_dim_decoder + encoder_dim)
            
            # Pass through GRU
            output, hidden = self.gru(gru_input, hidden)  # output: (batch, 1, hidden_dim_decoder)
            output = self.out(output.squeeze(1))  # shape: (batch, output_dim)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, target_len, output_dim)
        return outputs


class JointCTCAttentionModel(nn.Module):
    def __init__(self, hand_shape_dim, hand_pos_dim, lips_dim, visual_lips_dim, output_dim, hidden_dim=128):
        super(JointCTCAttentionModel, self).__init__()
        self.encoder = ThreeStreamFusionEncoder(hand_shape_dim, hand_pos_dim, lips_dim, visual_lips_dim, hidden_dim)
        self.attention_decoder = AttentionDecoder(encoder_dim=hidden_dim * 6, output_dim=output_dim)
        self.ctc_fc = nn.Linear(hidden_dim * 6, output_dim)
    
    def forward(self, hand_shape, hand_pos, lips, visual_lips, target_seq=None):
        encoder_out = self.encoder(hand_shape, hand_pos, lips, visual_lips)
        ctc_logits = self.ctc_fc(encoder_out)
        if target_seq is not None:
            attn_logits = self.attention_decoder(encoder_out, target_seq)
        else:
            attn_logits = None
        return ctc_logits, attn_logits

def joint_ctc_attention_loss(ctc_logits, attn_logits, target_seq, input_lengths, label_lengths, alpha, device):
    # CTC loss branch
    log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
    ctc_loss = F.ctc_loss(
        log_probs,
        target_seq,       # Padded labels (expected as LongTensor)
        input_lengths,
        label_lengths,
        blank=phoneme_to_index["<UNK>"],
    )
    
    if attn_logits is not None:
        # Attention branch loss
        attn_loss = F.cross_entropy(
            attn_logits.view(-1, attn_logits.size(-1)),
            target_seq.view(-1),
            ignore_index=phoneme_to_index["<UNK>"]
        )
        
        total_loss = alpha * ctc_loss + (1 - alpha) * attn_loss
        return total_loss, ctc_loss, attn_loss
    else:
        return ctc_loss, _, _


def validate_model(model, val_loader, alpha, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, batch_y in val_loader:
            # Ensure batch_y is of type long
            batch_X_hand_shape = batch_X_hand_shape.to(device)
            batch_X_hand_pos = batch_X_hand_pos.to(device)
            batch_X_lips = batch_X_lips.to(device)
            #batch_X_visual_lips = batch_X_visual_lips.to(device)  # Visual lips
            batch_y = batch_y.long().to(device)
            
            # Forward pass with teacher forcing
            ctc_logits, _ = model(
                batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, None, target_seq=None
            )
            # Compute input_lengths and label_lengths
            input_lengths = torch.full(
                (batch_X_hand_shape.size(0),),
                ctc_logits.size(1),
                dtype=torch.long,
                device=device
            )
            label_lengths = (batch_y != phoneme_to_index["<UNK>"]).sum(dim=1).to(device)
            
            # Compute combined loss
            loss, _, _ = joint_ctc_attention_loss(
                ctc_logits, None, batch_y, input_lengths, label_lengths, alpha, device
            )
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, avg_val_loss, 0


def train_model(model, train_loader, val_loader, optimizer, num_epochs, alpha, device):
    # Initialize variables to track the best validation PER
    best_val_per = float('inf') # Start with a very high value
    best_epoch = -1  # Track the epoch where the best model was saved
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_ctc_loss = 0.0
        epoch_attn_loss = 0.0
        
        for batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, batch_y in train_loader:
            # Convert targets to LongTensor and move data to device
            batch_X_hand_shape = batch_X_hand_shape.to(device)
            batch_X_hand_pos = batch_X_hand_pos.to(device)
            batch_X_lips = batch_X_lips.to(device)
            batch_y = batch_y.long().to(device)
            
            # Forward pass with teacher forcing
            ctc_logits, attn_logits = model(
                batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, None, target_seq=batch_y
            )
            
            # Compute input_lengths and label_lengths
            input_lengths = torch.full(
                (batch_X_hand_shape.size(0),),
                ctc_logits.size(1),
                dtype=torch.long,
                device=device
            )
            label_lengths = (batch_y != phoneme_to_index["<UNK>"]).sum(dim=1).to(device)
            
            # Compute joint loss
            loss, ctc_loss, attn_loss = joint_ctc_attention_loss(
                ctc_logits, attn_logits, batch_y, input_lengths, label_lengths, alpha, device
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_ctc_loss += ctc_loss.item()
            epoch_attn_loss += attn_loss.item()
        
        # Logging to W&B (or print)
        train_loss_avg = epoch_loss / len(train_loader)
        wandb.log({"train_loss": train_loss_avg, "epoch": epoch + 1})
        wandb.log({"train_ctc_loss": epoch_ctc_loss / len(train_loader), "epoch": epoch + 1})
        wandb.log({"train_attn_loss": epoch_attn_loss / len(train_loader), "epoch": epoch + 1})
        
        # Validate the model
        val_loss, val_ctc_loss, val_attn_loss = validate_model(model, val_loader, alpha, device)
        wandb.log({"val_loss": val_loss, "epoch": epoch + 1})
        wandb.log({"val_ctc_loss": val_ctc_loss, "epoch": epoch + 1})
        wandb.log({"val_attn_loss": val_attn_loss, "epoch": epoch + 1})
        
        # Optionally, decode and compute PER or WER on the validation set
        blank_token = phoneme_to_index["<UNK>"]
        decoded_val_sequences, true_val_sequences, decoded_val_gestures, true_val_gestures = decode_loader(
            model, val_loader, blank_token, index_to_phoneme, device, training=True
        )
        val_per = calculate_per_with_jiwer(decoded_val_sequences, true_val_sequences)
        val_gestures_per = calculate_per_with_jiwer(decoded_val_gestures, true_val_gestures)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {round(train_loss_avg, 3)}, "
                f"Val Loss: {round(val_loss, 3)}, Accuracy (1-PER): {round(1 - val_per, 3)}, "
                f"Accuracy gestures (1-PER): {round(1 - val_gestures_per, 3)}, "
                f"Time: {round(time.time() - epoch_start_time, 2)} sec")
        wandb.log({"val_per": 1 - val_per, "epoch": epoch + 1})
        wandb.log({"val_wer": val_per, "epoch": epoch + 1})
        # Save the model if it achieves the best validation PER
        #if val_per < best_val_per:
        #    best_val_per = val_per
        #    best_epoch = epoch + 1
        #    # Save the model checkpoint
        #    model_save_path = f"/pasteur/appa/homes/bsow/ACSR/src/acsr/saved_models/gridsearch/new_feat_acsr_{args.learning_rate}_{args.alpha}_{args.encoder_hidden_dim}_{args.n_layers}_{args.optimizer}.pt"
        #    optimizer_save_path = f"/pasteur/appa/homes/bsow/ACSR/src/acsr/saved_models/gridsearch/optim_new_feat_acsr_{args.learning_rate}_{args.alpha}_{args.encoder_hidden_dim}_{args.n_layers}_{args.optimizer}.pt"
        #    torch.save(model.state_dict(), model_save_path)
        #    torch.save(optimizer.state_dict(), optimizer_save_path)
        #    print(f"New best model saved at epoch {best_epoch} with Val PER: {round(best_val_per, 3)}")
        
        sys.stdout.flush()
    print(f"Training complete. Best model saved at epoch {best_epoch} with Val PER: {round(best_val_per, 3)}")

def syllables_to_gestures(syllable_sequence):
    """
    Convert a sequence of syllables into a sequence of gestures.
    
    Args:
        syllable_sequence (list): A list of syllables (strings).
        
    Returns:
        list: A list of gesture strings in the format "handshape-position".
    """
    gestures = []
    for syllable in syllable_sequence:
        if syllable == "<SOS>" or syllable == "<EOS>" or syllable == "<PAD>" or syllable == "<UNK>" or syllable == "_":
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
    return gestures

def syllables_to_phonemes(syllable_sequence):
    phonemes = []
    for syllable in syllable_sequence:
        if syllable == " ":
            phonemes.append(" ")
            continue
        if syllable == "<SOS>" or syllable == "<EOS>" or syllable == "<PAD>" or syllable == "<UNK>" or syllable == "_":
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
    
    return phonemes


def greedy_decoder(output, blank):
    arg_maxes = torch.argmax(output, dim=2)  # Get the most likely class for each time step
    decodes = []
    for args in arg_maxes:
        args = torch.unique_consecutive(args)  # Remove consecutive repeated indices
        decode = []
        for index in args:
            if index != blank:
                decode.append(index.item())  # Append non-blank and non-repeated tokens
        decodes.append(decode)
    return decodes


def decode_loader(model, loader, blank, index_to_phoneme, device='cuda', training=False):
    """
    Decode sequences from a DataLoader using the CTC branch of the joint model.
    This function now obtains two outputs from greedy_decoder: the raw per-timestep
    top-5 tokens (including blanks) and the collapsed (final) predictions.
    
    Args:
        model: The joint model.
        loader: DataLoader yielding batches.
        blank (int): The blank token index.
        index_to_phoneme (dict): Mapping from indices to phonemes.
        device (str): e.g., 'cuda' or 'cpu'.
        training (bool): If False, prints the decoded outputs.
    
    Returns:
        tuple: (all_collapsed_tokens, all_true_sequences, all_decoded_gestures, all_true_gestures)
               where all_collapsed_tokens is the list of collapsed decoded sequences.
    """
    model.eval()
    all_raw_decoded_sequences = []       # To store raw top-5 info per timestep
    all_collapsed_decoded_sequences = [] # To store final collapsed sequences
    all_true_sequences = []
    all_decoded_gestures = []
    all_true_gestures = []
    
    with torch.no_grad():
        for batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, batch_y in loader:
            batch_X_hand_shape = batch_X_hand_shape.to(device)
            batch_X_hand_pos = batch_X_hand_pos.to(device)
            batch_X_lips = batch_X_lips.to(device)
            batch_y = batch_y.to(device)
            
            ctc_logits, _ = model(batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, None)
            collapsed_decodes = greedy_decoder(ctc_logits, blank=blank)
            collapsed_decodes = [
                [index_to_phoneme[idx] for idx in sequence] 
                for sequence in collapsed_decodes
            ]
            decoded_gestures = [syllables_to_gestures(seq) for seq in collapsed_decodes]
            
            all_collapsed_decoded_sequences.extend(collapsed_decodes)
            all_decoded_gestures.extend(decoded_gestures)
            
            # Process true labels
            true_phoneme_sequences = []
            for sequence in batch_y:
                seq_phonemes = [
                    index_to_phoneme[idx.item()]
                    for idx in sequence 
                    if idx != blank and index_to_phoneme[idx.item()] != " "
                ]
                true_phoneme_sequences.append(seq_phonemes)
            all_true_sequences.extend(true_phoneme_sequences)
            all_true_gestures.extend([syllables_to_gestures(seq) for seq in true_phoneme_sequences])
    
    return all_collapsed_decoded_sequences, all_true_sequences, all_decoded_gestures, all_true_gestures


def calculate_per_with_jiwer(decoded_sequences, true_sequences):
    # Convert phoneme sequences to space-separated strings
    decoded_str = [" ".join([char for char in seq ]) for seq in decoded_sequences]
    true_str = [" ".join([char for char in seq]) for seq in true_sequences]
    # Calculate PER using jiwer
    try:
        per = jiwer.wer(true_str, decoded_str)
    except Exception as e:
        print("Error calculating PER:", e)
        print("True sequences:", true_str)
        print("Decoded sequences:", decoded_str)
    return per




if __name__ == "__main__":
    import argparse
    import argparse
    # Argument parsing
    parser = argparse.ArgumentParser(description="Grid search for model hyperparameters.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Weighting factor for CTC and attention losses")
    parser.add_argument("--encoder_hidden_dim", type=int, default=128, help="Hidden dimension for GRUs")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of GRU layers")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"], help="Optimizer to use")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()
    # Directories
    data_dir = r'/pasteur/appa/homes/bsow/ACSR/output/predictions'
    phoneme_dir_train = r'/pasteur/appa/homes/bsow/ACSR/data/training_videos/CSF22_train/new_labels'
    phoneme_dir_test = r'/pasteur/appa/homes/bsow/ACSR/data/training_videos/CSF22_test/new_labels'
    coordinates_dir_train = r'/pasteur/appa/homes/bsow/ACSR/output/extracted_coordinates_train'
    coordinates_dir_test = r'/pasteur/appa/homes/bsow/ACSR/output/extracted_coordinates_test'
    features_dir_train = r'/pasteur/appa/homes/bsow/ACSR/output/extracted_features_train'
    features_dir_test = r'/pasteur/appa/homes/bsow/ACSR/output/extracted_features_test'
    labels_dir_train = r'/pasteur/appa/homes/bsow/ACSR/data/training_videos/CSF22_train/new_labels'
    labels_dir_test = r'/pasteur/appa/homes/bsow/ACSR/data/training_videos/CSF22_test/new_labels'

    features_data_train = load_csv_files(features_dir_train, "_features")
    features_data_test = load_csv_files(features_dir_test, "_features")
    # Find phoneme files
    base_names_train = features_data_train.keys()
    base_names_test = list(features_data_test.keys())
    phoneme_files_train = find_phoneme_files(phoneme_dir_train, base_names_train)
    phoneme_files_test = find_phoneme_files(phoneme_dir_test, base_names_test)
    print("Number of phoneme files found in the train set:", len(phoneme_files_train))
    print("Number of phoneme files found in the test set:", len(phoneme_files_test))
    # Prepare data
    train_videos_data, syllable_counter = prepare_data_for_videos_no_sliding_windows(
        base_names_train, phoneme_files_train, features_dir_train, labels_dir_train, phoneme_to_index
    )

    test_videos_data, _ = prepare_data_for_videos_no_sliding_windows(
        base_names_test, phoneme_files_test, features_dir_test, labels_dir_test, phoneme_to_index
    )

    syllable_df = pd.DataFrame.from_dict(syllable_counter, orient='index', columns=['frequency'])
    syllable_df.index.name = 'syllable'
    syllable_df.reset_index(inplace=True)
    syllable_df = syllable_df.sort_values(by='frequency', ascending=False)

    # Save the syllable distribution to a CSV file
    output_csv_path = os.path.join("/pasteur/appa/homes/bsow/ACSR/src/acsr", 'syllable_distribution.csv')
    syllable_df.to_csv(output_csv_path, index=False)
    print(f"Syllable distribution saved to {output_csv_path}")


    # Final organized data
    train_data = {
        "X_acoustic_hand_shape": [train_videos_data[video]["X_acoustic_hand_shape"] for video in train_videos_data],  # Hand shape coordinates
        "X_acoustic_hand_pos": [train_videos_data[video]["X_acoustic_hand_pos"] for video in train_videos_data],      # Hand position coordinates
        "X_acoustic_lips": [train_videos_data[video]["X_acoustic_lips"] for video in train_videos_data],              # Lip coordinates
        "X_visual_lips": [], #[train_videos_data[video]["X_visual_lips"] for video in train_videos_data],                  # Visual lips
        "y": [train_videos_data[video]["y"] for video in train_videos_data],                                        # Phoneme labels
    }

    test_data = {
        "X_acoustic_hand_shape": [test_videos_data[video]["X_acoustic_hand_shape"] for video in test_videos_data],  # Hand shape coordinates
        "X_acoustic_hand_pos": [test_videos_data[video]["X_acoustic_hand_pos"] for video in test_videos_data],      # Hand position coordinates
        "X_acoustic_lips": [test_videos_data[video]["X_acoustic_lips"] for video in test_videos_data],              # Lip coordinates
        "X_visual_lips": [], #[test_videos_data[video]["X_visual_lips"] for video in test_videos_data],                  # Visual lips
        "y": [test_videos_data[video]["y"] for video in test_videos_data],                                        # Phoneme labels
    }

    # Split data
    #train_data, val_data = train_val_split(data)

    # Prepare DataLoaders
    train_loader = data_to_dataloader(train_data, batch_size=16, shuffle=True)
    val_loader = data_to_dataloader(test_data, batch_size=16, shuffle=False)

    print("Len of train dataset", len(train_data['X_acoustic_hand_shape']))
    print("Len of val dataset", len(test_data['X_acoustic_hand_shape']))

    # Check the DataLoader output
    for batch_X_acoustic_hand_shape, batch_X_acoustic_hand_pos, batch_X_acoustic_lips, batch_y in train_loader:
        print("Batch X_acoustic_hand_shape shape:", batch_X_acoustic_hand_shape.shape)
        print("Batch X_acoustic_hand_pos shape:", batch_X_acoustic_hand_pos.shape)
        print("Batch X_acoustic_lips shape:", batch_X_acoustic_lips.shape)
        #print("Batch X_visual_lips shape:", batch_X_visual_lips.shape)
        #print("Batch X_teacher shape:", batch_X_teacher.shape)
        print("Batch y shape:", batch_y.shape)
        print("Output dim of the model: ", len(phoneme_to_index))
        break

    # Model parameters
    learning_rate = 1e-3
    batch_size = 16
    epochs = 8000
    encoder_hidden_dim = 128
    output_dim = len(phoneme_to_index)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    level = "syllables"
    n_layers_gru = 2

    # Initialize wandb
    wandb.login(key="580ab03d7111ed25410f9831b06b544b5f5178a2")
    
    config = {
        "learning_rate": args.learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "encoder_hidden_dim": args.encoder_hidden_dim,
        "output_dim": output_dim,
        "n_layers_gru": args.n_layers,
        "alpha": args.alpha,
        "device": device,
        "level": level,
        "optimizer": args.optimizer,
    }
    wandb.init(project="acsr", config=config)
    
    print("CONFIG: ")
    for key in config:
        print("    ", key, ": ", config[key])
    print()
    # Define the model
    acoustic_model = JointCTCAttentionModel(
        hand_shape_dim=batch_X_acoustic_hand_shape.shape[-1],  # Number of hand shape keypoints
        hand_pos_dim=batch_X_acoustic_hand_pos.shape[-1],      # Number of hand position keypoints
        lips_dim=batch_X_acoustic_lips.shape[-1],              # Number of lip keypoints
        visual_lips_dim=None,        # Number of visual lip keypoints
        output_dim=output_dim,  # Number of phonemes
        hidden_dim=args.encoder_hidden_dim,  # Hidden dimension for GRUs
    )

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)
    acoustic_model.to(device)
    # load the trained acoustic model
    #acoustic_model.load_state_dict(torch.load("/pasteur/appa/homes/bsow/ACSR/src/acsr/saved_models/model_retrained_16_best_val_per.pt", map_location=device))
    #acoustic_model.load_state_dict(torch.load("/pasteur/appa/homes/bsow/ACSR/src/acsr/saved_models/gridsearch/new_feat_acsr_0.001_0.5_128_2_adam.pt", map_location=device))
    acoustic_model.to(device)
    
    if args.optimizer == "adam":
        optimizer = Adam(acoustic_model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    elif args.optimizer == "adamw":
        from torch.optim import AdamW
        optimizer = AdamW(acoustic_model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # Start training
    train_model(acoustic_model, train_loader, val_loader, num_epochs=epochs, alpha=args.alpha, device=device, optimizer=optimizer)


    blank_token =  phoneme_to_index["<UNK>"]
    #decoded_train_sequences, true_train_sequences, decoded_train_gestures, true_train_gestures = decode_loader(acoustic_model, train_loader, blank_token, index_to_phoneme, device, training=True)
    decoded_val_sequences, true_val_sequences, decoded_val_gestures, true_val_gestures = decode_loader(acoustic_model, val_loader, blank_token, index_to_phoneme, device, training=False)

    # Print results
    #print("Decoded training phoneme sequences:", decoded_train_sequences[:5])
    #print("True training phoneme sequences:", true_train_sequences[:5])
    print("Decoded validation phoneme sequences:", decoded_val_sequences[:5])
    print("True validation phoneme sequences:", true_val_sequences[:5])
    sys.stdout.flush()
    #train_per = calculate_per_with_jiwer(decoded_train_sequences, true_train_sequences)
    import sys
    import numpy as np


    # Initialize lists to store PER values
    val_per_list = []
    val_per_gestures_list = []

        
    val_per = calculate_per_with_jiwer(decoded_val_sequences, true_val_sequences)
    print("Validation PER (jiwer):", val_per, "1 - PER: ", 1 - val_per)
    #compute_and_print_wer_complements(decoded_val_sequences, true_val_sequences)
    print("Decoded validation gesture sequences:", decoded_val_gestures[:5])
    print("True validation gesture sequences:", true_val_gestures[:5])

    # Calculate PER for gestures
    val_per_gestures = calculate_per_with_jiwer(decoded_val_gestures, true_val_gestures)
    print("Validation PER (jiwer) for gestures:", val_per_gestures, "1 - PER: ", 1 - val_per_gestures)
    sys.stdout.flush()
    #compute_and_print_wer_complements(decoded_val_gestures, true_val_gestures)
    decoded_val_characters = [syllables_to_phonemes(seq) for seq in decoded_val_sequences]
    true_val_characters = [syllables_to_phonemes(seq) for seq in true_val_sequences]
    
    val_per_characters = calculate_per_with_jiwer(decoded_val_characters, true_val_characters)
    print("Validation PER (jiwer) for characters:", val_per_characters, "1 - PER: ", 1 - val_per_characters)
    #compute_and_print_wer_complements(decoded_val_characters, true_val_characters)
    print("="*190)


    # Initialize model
    #nextsyllable_model = NextSyllableLSTM(
    #    vocab_size=len(phoneme_to_index),
    #    embedding_dim=200,
    #    hidden_dim=512,
    #    num_layers=4,
    #    dropout=0.2
    #).to(device)
#   # # Load model weights
    #nextsyllable_model.load_state_dict(torch.load("/pasteur/appa/homes/bsow/ACSR/src/acsr/saved_models/best_syllable_model_def2.pth", map_location=device))
    ## Ensure both models are on the same device
    #nextsyllable_model.to(device)
    #nextsyllable_model.eval()
    ##
    ## After training your models, perform decoding
    #blank_token = phoneme_to_index["<UNK>"]
    #beam_width = 15
    #alpha = 0.5  # Adjust alpha to balance between models
    #decoded_val_sequences
#   # # Rescored decoding
    #rescored_sequences, _ = decode_loader_beam_vectorized(
    #    acoustic_model, nextsyllable_model, val_loader, blank_token, index_to_phoneme, 
    #    device=device, alpha=args.alpha, beta=args.beta
    #)
#   # # Calculate PER
    #greedy_per = calculate_per_with_jiwer(decoded_val_sequences, true_val_sequences)
    #rescored_per = calculate_per_with_jiwer(rescored_sequences, true_val_sequences)
####
    #phonemes_rescored = [syllables_to_phonemes(seq) for seq in rescored_sequences]
    #phoneme_rescored_per = calculate_per_with_jiwer(phonemes_rescored, true_val_characters)
    #rescored_gestures = [syllables_to_gestures(seq) for seq in rescored_sequences]
    #
    #        
    #print(f"Greedy 1 - PER: {1 - greedy_per:.3f}, Rescored 1 - PER: {1 - rescored_per:.3f}, Rescored phoneme 1 - PER: {1 - phoneme_rescored_per:.3f}")
####
    #rescored_gestures_per = calculate_per_with_jiwer(rescored_gestures, true_val_gestures)
    #print(f"Rescored gestures 1 - PER: {1 - rescored_gestures_per:.3f}")
    #
    #print("Decoded validation phoneme sequences:", rescored_sequences[:])
    #print("True validation phoneme sequences:", true_val_sequences[:])
    #
    #print()
    #print("Rescored gestures: ", rescored_gestures[:])
    #Finish W&B run
    wandb.finish()
    print("="*210)


    