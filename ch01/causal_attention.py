import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, context_len, dropout=0.1, use_bias=False):
        super().__init__()
        self.head_dim = head_dim
        
        # Projection layers
        self.query_proj = nn.Linear(embed_dim, head_dim, bias=use_bias)
        self.key_proj   = nn.Linear(embed_dim, head_dim, bias=use_bias)
        self.value_proj = nn.Linear(embed_dim, head_dim, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # Lower triangular mask (causal)
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(context_len, context_len))
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        queries = self.query_proj(x)
        keys    = self.key_proj(x)
        values  = self.value_proj(x)

        # Scaled dot-product attention
        # (batch, seq, head_dim) @ (batch, head_dim, seq) -> (batch, seq, seq)
        scores = torch.matmul(queries, keys.transpose(1, 2))
        scores = scores / (self.head_dim ** 0.5)

        # Apply causal mask (fill 0s with -inf)
        mask = self.mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Compute final context vectors
        context_vectors = torch.matmul(weights, values)
        return context_vectors

# Example usage:
# model = CausalAttention(embed_dim=256, head_dim=64, context_len=1024)
