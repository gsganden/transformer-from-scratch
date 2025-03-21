import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, BoolTensor

# Try to import Flash Attention
FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    # Flash Attention is not available
    pass


def _get_device():
    device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device_str)


def distribute_over_heads(mat: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reshape from `batch_size, sequence_length, num_heads x head_dim` to
    `batch_size, num_heads, sequence_length, head_dim`
    """
    batch_size, sequence_length, combined_head_dim = mat.shape
    head_dim = combined_head_dim // num_heads
    return mat.view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)


def consolidate_over_heads(mat: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reshape from `batch_size, num_heads, sequence_length, head_dim` to
    `batch_size, sequence_length, num_heads x head_dim`
    """
    batch_size, num_heads, sequence_length, head_dim = mat.shape
    combined_head_dim = head_dim * num_heads
    return mat.transpose(1, 2).reshape(batch_size, sequence_length, combined_head_dim)


def generate_causal_mask(sequence_length: int) -> torch.Tensor:
    return (
        (
            torch.tril(
                torch.ones(
                    (
                        sequence_length,
                        sequence_length,
                    ),
                ),
            )
        )
        .bool()
        .to(_get_device())
    )


class EagerBidirectionalAttentionBlock(nn.Module):
    """
    Attention block implementing multi-head bidirectional (full) attention using
    the eager approach.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """
        Initialize the bidirectional attention block with eager implementation.

        Args:
            hidden_dim: Dimension of the input and output features
            num_heads: Number of attention heads
            dropout: Output dropout probability (0.0 means no dropout)

        Note:
            - Make sure to check that hidden_dim is divisible by num_heads
            - You'll need to create linear (projection) layers for query, key, and value
            - Don't forget the output linear (projection) layer
            - Create an output dropout layer
        """
        super().__init__()

        if not hidden_dim % num_heads == 0:
            raise ValueError(f"{hidden_dim=} not divisible by {num_heads=}")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.head_dim = hidden_dim // num_heads

        self.Wq = nn.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            bias=True,
        )
        self.Wk = nn.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            bias=True,
        )
        self.Wv = nn.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            bias=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.Wo = nn.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            bias=True,
        )

    def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].
            mask: Optional boolean mask of shape [batch_size, sequence_length]
                  where True indicates attended tokens and False masked positions

        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim] after attention.
        """
        q = distribute_over_heads(self.Wq(x), self.num_heads)
        k = distribute_over_heads(self.Wk(x), self.num_heads)
        v = distribute_over_heads(self.Wv(x), self.num_heads)

        attn_scores = q @ k.transpose(-2, -1)
        if mask is not None:
            # repeat mask along num_heads and the first num_tokens dimension
            # so that each token ignores the specified tokens
            attn_scores.masked_fill_(~mask[:, None, None] if mask.dim() == 2 else ~mask[:, None], -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        return self.dropout(self.Wo(consolidate_over_heads(attn_weights @ v, self.num_heads)))


class EagerCausalAttentionBlock(EagerBidirectionalAttentionBlock):
    """
    Attention block implementing multi-head causal (masked) attention using
    the eager approach.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 128,
    ):
        """
        Initialize the causal attention block with eager implementation.

        Args:
            hidden_dim: Dimension of the input and output features
            num_heads: Number of attention heads
            dropout: Output dropout probability (0.0 means no dropout)
            max_seq_len: Maximum sequence length (for causal masking)
        Note:
            - Make sure to check that hidden_dim is divisible by num_heads
            - You'll need to create linear (projection) layers for query, key, and value
            - Don't forget the output linear (projection) layer
            - Create an output dropout layer
        """
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.register_buffer(
            "causal_mask",
            generate_causal_mask(
                sequence_length=max_seq_len,
            )
        )

    def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].
            mask: Optional boolean mask of shape [batch_size, sequence_length]
                  where True indicates attended tokens and False masked positions

        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim] after attention.
        """
        seq_len = x.shape[1]
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        return super().forward(
            x=x,
            mask=(
                causal_mask[None] if mask is None
                else mask[:, None] & causal_mask
            )
        )


class SDPABidirectionalAttentionBlock(EagerBidirectionalAttentionBlock):
    """
    Attention block implementing multi-head bidirectional (full) attention using
    PyTorch's scaled_dot_product_attention (SDPA).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """
        Initialize the bidirectional attention block with SDPA implementation.

        Args:
            hidden_dim: Dimension of the input and output features
            num_heads: Number of attention heads
            dropout: Output dropout probability (0.0 means no dropout)

        Note:
            - Make sure to check that hidden_dim is divisible by num_heads
            - You'll need to create linear (projection) layers for query, key, and value
            - Don't forget the output linear (projection) layer
            - Create an output dropout layer
        """
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].
            mask: Optional boolean mask of shape [batch_size, sequence_length]
                  where True indicates attended tokens and False masked positions

        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim] after attention.
        """
        return self.dropout(
            self.Wo(
                consolidate_over_heads(
                    F.scaled_dot_product_attention(
                        query=distribute_over_heads(self.Wq(x), num_heads=self.num_heads),
                        key=distribute_over_heads(self.Wk(x), num_heads=self.num_heads),
                        value=distribute_over_heads(self.Wv(x), num_heads=self.num_heads),
                        attn_mask=mask if mask is None else mask[:, None, None] if mask.dim() == 2 else mask[:, None],
                    ),
                    num_heads=self.num_heads,
                )
            )
        )


class SDPACausalAttentionBlock(SDPABidirectionalAttentionBlock):
    """
    Attention block implementing multi-head causal (masked) attention using
    PyTorch's scaled_dot_product_attention (SDPA).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """
        Initialize the causal attention block with SDPA implementation.

        Args:
            hidden_dim: Dimension of the input and output features
            num_heads: Number of attention heads
            dropout: Output dropout probability (0.0 means no dropout)

        Note:
            - Make sure to check that hidden_dim is divisible by num_heads
            - You'll need to create linear (projection) layers for query, key, and value
            - Don't forget the output linear (projection) layer
            - Create an output dropout layer
        """
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].
            mask: Optional boolean mask of shape [batch_size, sequence_length]
                  where True indicates attended tokens and False masked positions

        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim] after attention.
        """
        causal_mask = generate_causal_mask(x.shape[1])[None]
        mask = causal_mask if mask is None else causal_mask & mask[:, None]
        return super().forward(
            x,
            mask=mask,
        )


class FlashBidirectionalAttentionBlock(nn.Module):
    """
    Attention block implementing multi-head bidirectional (full) attention using
    Flash Attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """
        Initialize the bidirectional attention block with Flash Attention implementation.

        Args:
            hidden_dim: Dimension of the input and output features
            num_heads: Number of attention heads
            dropout: Output dropout probability (0.0 means no dropout)

        Note:
            - Make sure to check that hidden_dim is divisible by num_heads
            - Check if Flash Attention is available (FLASH_ATTN_AVAILABLE)
            - You'll need to create linear (projection) layers for query, key, and value
            - Don't forget the output linear (projection) layer
            - Create an output dropout layer
        """
        super().__init__()
        raise NotImplementedError("Implement initialization for bidirectional attention block using Flash Attention")

    def forward(self, x: Tensor, cu_seqlens: Tensor, max_seqlen: int) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [total_seq_len, hidden_dim].
            cu_seqlens: Cumulative sequence lengths tensor of shape [batch_size + 1]
                    Used instead of an attention mask for both masking and
                    variable-length sequences. Example:
                        cu_seqlens = torch.tensor([0, 10, 30, 60])
                    This means there are three sequences in the batch:
                        - First sequence has 10 tokens
                        - Second sequence has 20 tokens
                        - Third sequence has 30 tokens
            max_seqlen: Maximum sequence length in the batch. In the example above,
                        the maximum sequence length is 30.

        Returns:
            Tensor of shape [total_seq_len, hidden_dim] after attention.
        """
        raise NotImplementedError("Implement bidirectional attention block using Flash Attention")


class FlashCausalAttentionBlock(nn.Module):
    """
    Attention block implementing multi-head causal (masked) attention using
    Flash Attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """
        Initialize the causal attention block with Flash Attention implementation.

        Args:
            hidden_dim: Dimension of the input and output features
            num_heads: Number of attention heads
            dropout: Output dropout probability (0.0 means no dropout)

        Note:
            - Make sure to check that hidden_dim is divisible by num_heads
            - Check if Flash Attention is available (FLASH_ATTN_AVAILABLE)
            - You'll need to create linear (projection) layers for query, key, and value
            - Don't forget the output linear (projection) layer
            - Create an output dropout layer
        """
        super().__init__()
        raise NotImplementedError("Implement initialization for causal attention block using Flash Attention")

    def forward(self, x: Tensor, cu_seqlens: Tensor, max_seqlen: int) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [total_seq_len, hidden_dim].
            cu_seqlens: Cumulative sequence lengths tensor of shape [batch_size + 1]
                    Used instead of an attention mask for both masking and
                    variable-length sequences. Example:
                        cu_seqlens = torch.tensor([0, 10, 30, 60])
                    This means there are three sequences in the batch:
                        - First sequence has 10 tokens
                        - Second sequence has 20 tokens
                        - Third sequence has 30 tokens
            max_seqlen: Maximum sequence length in the batch. In the example above,
                        the maximum sequence length is 30.

        Returns:
            Tensor of shape [total_seq_len, hidden_dim] after attention.
        """
        raise NotImplementedError("Implement causal attention block using Flash Attention")
