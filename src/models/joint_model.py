import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .encoder import ThreeStreamFusionEncoder
from .decoder import AttentionDecoder

from utils.text_processing import load_vocabulary

class JointCTCAttentionModel(nn.Module):
    """Joint CTC-Attention model for sequence recognition."""
    
    def __init__(self, hand_shape_dim: int, hand_pos_dim: int, lips_dim: int,
                 output_dim: int, hidden_dim: int = 128, n_layers: int = 2):
        """
        Initialize the joint CTC-Attention model.
        
        Args:
            hand_shape_dim: Dimension of hand shape features
            hand_pos_dim: Dimension of hand position features
            lips_dim: Dimension of lip features
            output_dim: Dimension of output vocabulary
            hidden_dim: Hidden dimension for model layers
            n_layers: Number of layers for GRUs
        """
        super().__init__()
        
        # Encoder
        self.encoder = ThreeStreamFusionEncoder(
            hand_shape_dim=hand_shape_dim,
            hand_pos_dim=hand_pos_dim,
            lips_dim=lips_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers
        )
        
        # Decoder
        encoder_output_dim = hidden_dim * 6  # 3 streams * 2 (bidirectional)
        self.attention_decoder = AttentionDecoder(
            encoder_dim=encoder_output_dim,
            output_dim=output_dim
        )
        
        # CTC output projection
        self.ctc_fc = nn.Linear(encoder_output_dim, output_dim)
        
        # Store blank index
        vocab = load_vocabulary('/pasteur/appa/homes/bsow/ACSR/data/french_dataset/vocab.txt')
        self.blank_idx = next((idx for token, idx in vocab.items() if token == '<UNK>'), None)
    
    def forward(self, hand_shape: torch.Tensor, hand_pos: torch.Tensor,
                lips: torch.Tensor, target_seq: Optional[torch.Tensor] = None,
                max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            hand_shape: Hand shape features of shape (batch_size, seq_length, hand_shape_dim)
            hand_pos: Hand position features of shape (batch_size, seq_length, hand_pos_dim)
            lips: Lip features of shape (batch_size, seq_length, lips_dim)
            target_seq: Target sequence for training of shape (batch_size, seq_length)
            max_len: Maximum sequence length for inference
            
        Returns:
            Tuple of (CTC logits, Attention logits)
        """
        # Encode input features
        encoder_outputs = self.encoder(hand_shape, hand_pos, lips)
        
        # CTC branch
        ctc_logits = self.ctc_fc(encoder_outputs)
        
        # Attention branch - explicitly pass target_seq (which may be None) and max_len
        if target_seq is not None:
            attn_logits = self.attention_decoder(encoder_outputs, target_seq)
        else:
            attn_logits = None
        return ctc_logits, attn_logits
    
    def compute_loss(self, ctc_logits: torch.Tensor, attn_logits: torch.Tensor,
                    target_seq: torch.Tensor, input_lengths: torch.Tensor,
                    label_lengths: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        """
        Compute the joint CTC-Attention loss.
        
        Args:
            ctc_logits: CTC branch logits
            attn_logits: Attention branch logits
            target_seq: Target sequence
            input_lengths: Input sequence lengths
            label_lengths: Label sequence lengths
            alpha: Weight for balancing CTC and Attention losses
            
        Returns:
            Combined loss value
        """
        # CTC loss
        log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
        ctc_loss = F.ctc_loss(
            log_probs,
            target_seq,
            input_lengths,
            label_lengths,
            blank=self.blank_idx
        )
        
        # Attention loss (cross entropy)
        attn_loss = F.cross_entropy(
            attn_logits.view(-1, attn_logits.size(-1)),
            target_seq.view(-1),
            ignore_index=self.blank_idx
        )
        
        # Combined loss
        loss = alpha * ctc_loss + (1 - alpha) * attn_loss
        
        return loss 