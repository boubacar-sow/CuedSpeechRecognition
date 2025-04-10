import torch
import torch.nn as nn

class ThreeStreamFusionEncoder(nn.Module):
    """Three-stream fusion encoder for processing hand shape, hand position, and lip features."""
    
    def __init__(self, hand_shape_dim: int, hand_pos_dim: int, lips_dim: int, 
                 hidden_dim: int = 128, n_layers: int = 2):
        """
        Initialize the three-stream fusion encoder.
        
        Args:
            hand_shape_dim: Dimension of hand shape features
            hand_pos_dim: Dimension of hand position features
            lips_dim: Dimension of lip features
            hidden_dim: Hidden dimension for GRU layers
            n_layers: Number of GRU layers
        """
        super().__init__()
        
        # Individual stream processors
        self.hand_shape_gru = nn.GRU(
            input_size=hand_shape_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True
        )
        
        self.hand_pos_gru = nn.GRU(
            input_size=hand_pos_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True
        )
        
        self.lips_gru = nn.GRU(
            input_size=lips_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Fusion layer
        fusion_input_dim = hidden_dim * 6  # 3 streams * 2 (bidirectional)
        self.fusion_gru = nn.GRU(
            input_size=fusion_input_dim,
            hidden_size=hidden_dim * 3,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, hand_shape: torch.Tensor, hand_pos: torch.Tensor, 
                lips: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.
        
        Args:
            hand_shape: Hand shape features of shape (batch_size, seq_length, hand_shape_dim)
            hand_pos: Hand position features of shape (batch_size, seq_length, hand_pos_dim)
            lips: Lip features of shape (batch_size, seq_length, lips_dim)
            
        Returns:
            Encoded features of shape (batch_size, seq_length, hidden_dim * 6)
        """
        # Process each stream
        hand_shape_out, _ = self.hand_shape_gru(hand_shape)
        hand_pos_out, _ = self.hand_pos_gru(hand_pos)
        lips_out, _ = self.lips_gru(lips)
        
        # Concatenate all streams
        combined = torch.cat([hand_shape_out, hand_pos_out, lips_out], dim=-1)
        
        # Fusion
        fused, _ = self.fusion_gru(combined)
        
        return fused 