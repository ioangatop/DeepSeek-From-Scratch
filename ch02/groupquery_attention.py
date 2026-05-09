from torch import nn
import torch


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        context_length: int,
        n_heads: int,
        num_groups: int,
        dropout: float = 0.1,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()

        self._embed_dim = embed_dim
        self._output_dim = output_dim
        self._context_length = context_length
        self._n_heads = n_heads
        self._head_dim = self._output_dim // self._n_heads
        self._num_groups = num_groups

        self._q_projection = nn.Linear(self._embed_dim, self._output_dim, bias=qkv_bias)
        self._k_projection = nn.Linear(self._embed_dim, self._num_groups * self._head_dim, bias=qkv_bias)
        self._v_projection = nn.Linear(self._embed_dim, self._num_groups * self._head_dim, bias=qkv_bias)
        self._o_projection = nn.Linear(self._output_dim, self._output_dim, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length))
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape

        queries = self._q_projection(tensor)
        keys = self._k_projection(tensor)

        queries = queries.view(batch_size, seq_len, self._n_heads, self._head_dim)
        keys = keys.view(batch_size, seq_len, self._num_groups, self._head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)

        keys = keys.repeat_interleave(self._n_heads // self._num_groups, dim=1)

        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / (self._head_dim ** 0.5)

        mask = self.mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        values = self._v_projection(tensor)
        values = values.view(batch_size, seq_len, self._num_groups, self._head_dim).transpose(1, 2)
        values = values.repeat_interleave(self._n_heads // self._num_groups, dim=1)

        context_vectors = torch.matmul(weights, values) 
        context_vectors = context_vectors.transpose(1, 2).contiguous()
        context_vectors = context_vectors.reshape(batch_size, seq_len, self._output_dim)
        return self._o_projection(context_vectors)
