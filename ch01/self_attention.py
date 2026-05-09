from torch import nn
import torch


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        context_length: int,
        dropout: float = 0.1,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()

        self._embed_dim = embed_dim
        self._output_dim = output_dim
        self._context_length = context_length

        self._q_projection = nn.Linear(self._embed_dim, self._output_dim, bias=qkv_bias)
        self._k_projection = nn.Linear(self._embed_dim, self._output_dim, bias=qkv_bias)
        self._v_projection = nn.Linear(self._embed_dim, self._output_dim, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape

        # (batch_size, seq_len, hidden_dim)
        queries = self._q_projection(tensor)
        keys = self._k_projection(tensor)

        # (batch_size, seq_len, seq_len)
        scores = torch.matmul(queries, keys.transpose(1, 2))
        scores = scores / (self._output_dim ** 0.5)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        values = self._v_projection(tensor)
        context_vectors = torch.matmul(weights, values)
        return context_vectors
