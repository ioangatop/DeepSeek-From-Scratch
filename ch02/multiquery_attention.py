from torch import nn
import torch


class MultiQueryAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        context_length: int,
        n_heads: int,
        dropout: float = 0.1,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()

        self._embed_dim = embed_dim
        self._output_dim = output_dim
        self._context_length = context_length
        self._n_heads = n_heads
        self._head_dim = self._output_dim // self._n_heads

        self._q_projection = nn.Linear(self._embed_dim, self._output_dim, bias=qkv_bias)
        self._k_projection = nn.Linear(self._embed_dim, self._head_dim, bias=qkv_bias)
        self._v_projection = nn.Linear(self._embed_dim, self._head_dim, bias=qkv_bias)
        self._o_projection = nn.Linear(self._output_dim, self._output_dim, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length))
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape

        # (batch_size, seq_len, hidden_dim)
        queries = self._q_projection(tensor)
        keys = self._k_projection(tensor)

        # (batch_size, seq_len, num_heads, head_dim)
        queries = queries.view(batch_size, seq_len, self._n_heads, self._head_dim)
        keys = keys.view(batch_size, seq_len, 1, self._head_dim)

        # (batch_size, num_heads, seq_len, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)

        # (batch_size, num_heads, seq_len, seq_len)
        # -> broadcasting handles (B, H, S, D) @ (B, 1, D, S)
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / (self._head_dim ** 0.5)

        # apply mask
        mask = self.mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # (batch_size, num_heads, seq_len, head_dim)
        values = self._v_projection(tensor)
        values = values.view(batch_size, seq_len, 1, self._head_dim).transpose(1, 2)

        # (batch_size, seq_len, output_dim)
        context_vectors = torch.matmul(weights, values) # Broadcasting handles H vs 1
        context_vectors = context_vectors.transpose(1, 2).contiguous()
        context_vectors = context_vectors.reshape(batch_size, seq_len, self._output_dim)
        return self._o_projection(context_vectors)
