import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionDecoder(nn.Module):
    """Attention-based decoder for sequence generation."""
    
    def __init__(self, encoder_dim: int, output_dim: int, 
                 hidden_dim: int = None, n_layers: int = 1):
        """
        Initialize the attention decoder.
        
        Args:
            encoder_dim: Dimension of encoder outputs
            output_dim: Dimension of output vocabulary
            hidden_dim: Hidden dimension for decoder (defaults to encoder_dim)
            n_layers: Number of decoder layers
        """
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else encoder_dim
        
        # Embedding layer to convert token indices to embeddings
        self.embedding = nn.Embedding(output_dim, self.hidden_dim)
        
        # Decoder GRU
        self.gru = nn.GRU(
            input_size=self.encoder_dim * 2,  # Concatenated embedding and context
            hidden_size=self.hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        
        # Output projection
        self.out = nn.Linear(self.hidden_dim, output_dim)
        
    
    def forward(self, encoder_outputs: torch.Tensor, target_seq: torch.Tensor = None
                ) -> torch.Tensor:
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