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
    t = (
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
    print(f"In generate_causal_mask: {t=}")
    return t


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
        print(f"In __init__: {self.causal_mask=}")

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
        print(f"In forward: {self.causal_mask=}")

        return super().forward(
            x=x,
            mask=(
                causal_mask[None] if mask is None
                else mask[:, None] & causal_mask
            )
        )


class SDPABidirectionalAttentionBlock(nn.Module):
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
        super().__init__()
        raise NotImplementedError("Implement initialization for bidirectional attention block using PyTorch's SDPA")

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
        raise NotImplementedError("Implement bidirectional attention block using PyTorch's SDPA")


class SDPACausalAttentionBlock(nn.Module):
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
        super().__init__()
        raise NotImplementedError("Implement initialization for causal attention block using PyTorch's SDPA")

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
        raise NotImplementedError("Implement causal attention block using PyTorch's SDPA")


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
#         raise NotImplementedError("Implement causal attention block using Flash Attention")

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor, BoolTensor

# # Try to import Flash Attention
# FLASH_ATTN_AVAILABLE = False
# try:
#     from flash_attn import flash_attn_varlen_func

#     FLASH_ATTN_AVAILABLE = True
# except ImportError:
#     # Flash Attention is not available
#     pass


# class EagerBidirectionalAttentionBlock(nn.Module):
#     """
#     Attention block implementing multi-head bidirectional (full) attention using
#     the eager approach.
#     """

#     def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
#         super().__init__()
#         assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = hidden_dim // num_heads

#         # Projection layers for query, key, and value
#         self.Wq = nn.Linear(hidden_dim, hidden_dim)
#         self.Wk = nn.Linear(hidden_dim, hidden_dim)
#         self.Wv = nn.Linear(hidden_dim, hidden_dim)

#         # Output projection layer
#         self.Wo = nn.Linear(hidden_dim, hidden_dim)

#         self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

#     def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
#         """
#         Forward pass.

#         Args:
#             x: Input tensor of shape [batch_size, seq_len, hidden_dim].
#             mask: Optional boolean mask of shape [batch_size, sequence_length]
#                   where True indicates attended tokens and False masked positions

#         Returns:
#             Tensor of shape [batch_size, seq_len, hidden_dim] after attention.
#         """
#         batch_size, seq_len, hidden_dim = x.size()

#         # Compute Q, K, V projections
#         q = self.Wq(x)
#         k = self.Wk(x)
#         v = self.Wv(x)

#         # Reshape to separate the heads
#         # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_heads, head_dim]
#         q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

#         # Transpose to get [batch_size, num_heads, seq_len, head_dim]
#         q = q.transpose(1, 2)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)

#         # Compute attention scores: [batch_size, num_heads, seq_len, seq_len]
#         attn = q @ k.transpose(-2, -1)

#         # Scale by square root of head dimension
#         attn = attn / math.sqrt(self.head_dim)

#         # Apply mask if provided
#         if mask is not None:
#             # Reshape mask to [batch_size, 1, 1, seq_len]
#             # And invert since we want to mask out the positions where it is False
#             mask = ~mask.view(batch_size, 1, 1, seq_len)
#             attn = attn.masked_fill(mask, float("-inf"))

#         # Apply softmax to get attention weights
#         attn = F.softmax(attn, dim=-1)

#         # Apply attention weights to values
#         # [batch_size, num_heads, seq_len, seq_len] @ [batch_size, num_heads, seq_len, head_dim]
#         # -> [batch_size, num_heads, seq_len, head_dim]
#         output = attn @ v

#         # Transpose and reshape back to [batch_size, seq_len, hidden_dim]
#         output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

#         # Final projection
#         output = self.Wo(output)

#         # Apply dropout
#         output = self.dropout(output)
#         return output


# class EagerCausalAttentionBlock(nn.Module):
#     """
#     Attention block implementing multi-head causal (masked) attention using
#     the eager approach.
#     """

#     def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, max_seq_len: int = 1024):
#         super().__init__()
#         assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = hidden_dim // num_heads

#         # Projection layers for Q, K, V
#         self.Wq = nn.Linear(hidden_dim, hidden_dim)
#         self.Wk = nn.Linear(hidden_dim, hidden_dim)
#         self.Wv = nn.Linear(hidden_dim, hidden_dim)

#         # Output projection layer
#         self.Wo = nn.Linear(hidden_dim, hidden_dim)

#         self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

#         # Causal mask (upper triangular), here true means mask out the position so we don't need to invert it
#         self.register_buffer("causal_mask", torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1))

#     def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
#         """
#         Forward pass.

#         Args:
#             x: Input tensor of shape [batch_size, seq_len, hidden_dim].
#             mask: Optional boolean mask of shape [batch_size, sequence_length]
#                   where True indicates attended tokens and False masked positions

#         Returns:
#             Tensor of shape [batch_size, seq_len, hidden_dim] after attention.
#         """
#         batch_size, seq_len, hidden_dim = x.size()

#         # Compute projections for Q, K, V.
#         q = self.Wq(x)
#         k = self.Wk(x)
#         v = self.Wv(x)

#         # Reshape to separate the heads
#         # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_heads, head_dim]
#         q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

#         # Transpose to get [batch_size, num_heads, seq_len, head_dim]
#         q = q.transpose(1, 2)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)

#         # Compute attention scores: [batch_size, num_heads, seq_len, seq_len]
#         attn = q @ k.transpose(-2, -1)

#         # Scale by square root of head dimension
#         attn = attn / math.sqrt(self.head_dim)

#         # Create causal mask (upper triangular), here true means mask out the position so we don't need to invert it
#         # We need to slice the causal mask to the sequence length from the max sequence length
#         causal_mask = self.causal_mask[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)

#         # Apply causal mask
#         attn = attn.masked_fill(causal_mask, float("-inf"))

#         # Apply additional mask if provided
#         if mask is not None:
#             # Reshape mask to [batch_size, 1, 1, seq_len]
#             # And invert since we want to mask out the positions where it is False
#             mask = ~mask.view(batch_size, 1, 1, seq_len)
#             attn = attn.masked_fill(mask, float("-inf"))

#         # Apply softmax to get attention weights
#         attn = F.softmax(attn, dim=-1)

#         # Apply attention weights to values
#         # [batch_size, num_heads, seq_len, seq_len] @ [batch_size, num_heads, seq_len, head_dim]
#         # -> [batch_size, num_heads, seq_len, head_dim]
#         output = attn @ v

#         # Transpose and reshape back to [batch_size, seq_len, hidden_dim]
#         output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

#         # Final projection.
#         output = self.Wo(output)

#         # Apply dropout
#         output = self.dropout(output)
#         return output


# class SDPABidirectionalAttentionBlock(nn.Module):
#     """
#     Attention block implementing multi-head bidirectional (full) attention using
#     PyTorch's scaled_dot_product_attention (SDPA).
#     """

#     def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
#         super().__init__()
#         assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = hidden_dim // num_heads

#         # Projection layers.
#         self.Wq = nn.Linear(hidden_dim, hidden_dim)
#         self.Wk = nn.Linear(hidden_dim, hidden_dim)
#         self.Wv = nn.Linear(hidden_dim, hidden_dim)

#         # Output projection.
#         self.Wo = nn.Linear(hidden_dim, hidden_dim)

#         self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

#     def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
#         """
#         Forward pass.

#         Args:
#             x: Input tensor of shape [batch_size, seq_len, hidden_dim].
#             mask: Optional boolean mask of shape [batch_size, sequence_length]
#                   where True indicates attended tokens and False masked positions

#         Returns:
#             Tensor of shape [batch_size, seq_len, hidden_dim] after attention.
#         """
#         batch_size, seq_len, hidden_dim = x.size()

#         # Compute Q, K, V.
#         q = self.Wq(x)
#         k = self.Wk(x)
#         v = self.Wv(x)

#         # Reshape to separate the heads
#         # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_heads, head_dim]
#         q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

#         # Transpose to get [batch_size, num_heads, seq_len, head_dim]
#         q = q.transpose(1, 2)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)

#         # Prepare attention mask if provided
#         attn_mask = None
#         if mask is not None:
#             # Reshape to [batch_size, 1, 1, seq_len] for broadcasting
#             attn_mask = mask.view(batch_size, 1, 1, seq_len)

#         # Use PyTorch's scaled_dot_product_attention
#         output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

#         # Transpose and reshape back to [batch_size, seq_len, hidden_dim]
#         output = output.transpose(1, 2).view(batch_size, seq_len, hidden_dim)

#         # Final output projection.
#         output = self.Wo(output)

#         # Apply dropout
#         output = self.dropout(output)
#         return output


# class SDPACausalAttentionBlock(nn.Module):
#     """
#     Attention block implementing multi-head causal (masked) attention using
#     PyTorch's scaled_dot_product_attention (SDPA).
#     """

#     def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
#         super().__init__()
#         assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = hidden_dim // num_heads

#         # Projection layers.
#         self.Wq = nn.Linear(hidden_dim, hidden_dim)
#         self.Wk = nn.Linear(hidden_dim, hidden_dim)
#         self.Wv = nn.Linear(hidden_dim, hidden_dim)

#         # Output projection.
#         self.Wo = nn.Linear(hidden_dim, hidden_dim)

#         # Optional dropout.
#         self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

#     def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
#         """
#         Forward pass.

#         Args:
#             x: Input tensor of shape [batch_size, seq_len, hidden_dim].
#             mask: Optional boolean mask of shape [batch_size, sequence_length]
#                   where True indicates attended tokens and False masked positions

#         Returns:
#             Tensor of shape [batch_size, seq_len, hidden_dim] after attention.
#         """
#         batch_size, seq_len, hidden_dim = x.size()

#         # Compute Q, K, V.
#         q = self.Wq(x)
#         k = self.Wk(x)
#         v = self.Wv(x)

#         # Reshape to separate the heads
#         # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_heads, head_dim]
#         q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

#         # Transpose to get [batch_size, num_heads, seq_len, head_dim]
#         q = q.transpose(1, 2)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)

#         # Prepare attention mask if provided
#         attn_mask = None
#         if mask is not None:
#             # Reshape to [batch_size, 1, 1, seq_len] for broadcasting
#             attn_mask = mask.view(batch_size, 1, 1, seq_len)

#         # Use PyTorch's scaled_dot_product_attention
#         output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=True)

#         # Transpose and reshape back to [batch_size, seq_len, hidden_dim]
#         output = output.transpose(1, 2).view(batch_size, seq_len, hidden_dim)

#         # Final projection.
#         output = self.Wo(output)

#         # Apply dropout
#         output = self.dropout(output)
#         return output


# class FlashBidirectionalAttentionBlock(nn.Module):
#     """
#     Attention block implementing multi-head bidirectional (full) attention using
#     Flash Attention.
#     """

#     def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
#         super().__init__()
#         if not FLASH_ATTN_AVAILABLE:
#             raise ImportError("Flash Attention is not available. Install flash-attn package or choose another approach.")

#         assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.head_dim = hidden_dim // num_heads

#         # Projection layers.
#         self.Wq = nn.Linear(hidden_dim, hidden_dim)
#         self.Wk = nn.Linear(hidden_dim, hidden_dim)
#         self.Wv = nn.Linear(hidden_dim, hidden_dim)

#         # Output projection.
#         self.Wo = nn.Linear(hidden_dim, hidden_dim)

#         # Optional dropout.
#         self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

#     def forward(self, x: Tensor, cu_seqlens: Tensor, max_seqlen: int) -> Tensor:
#         """
#         Forward pass.

#         Args:
#             x: Input tensor of shape [total_seq_len, hidden_dim].
#             cu_seqlens: Cumulative sequence lengths tensor of shape [batch_size + 1]
#                         Used for variable sequence length batching in Flash Attention
#             max_seqlen: Maximum sequence length in the batch

#         Returns:
#             Tensor of shape [total_seq_len, hidden_dim] after attention.
#         """
#         # Get total sequence length and hidden dimension
#         total_seq_len, hidden_dim = x.shape

#         # Compute Q, K, V.
#         q = self.Wq(x)
#         k = self.Wk(x)
#         v = self.Wv(x)

#         # Reshape to separate the heads: [total_seq_len, hidden_dim] -> [total_seq_len, num_heads, head_dim]
#         q = q.view(total_seq_len, self.num_heads, self.head_dim)
#         k = k.view(total_seq_len, self.num_heads, self.head_dim)
#         v = v.view(total_seq_len, self.num_heads, self.head_dim)

#         # Use Flash Attention with variable sequence lengths
#         attn_output = flash_attn_varlen_func(
#             q,
#             k,
#             v,
#             cu_seqlens_q=cu_seqlens,
#             cu_seqlens_k=cu_seqlens,
#             max_seqlen_q=max_seqlen,
#             max_seqlen_k=max_seqlen,
#             causal=False,
#         )

#         # Reshape back to original format: [total_seq_len, num_heads, head_dim] -> [total_seq_len, hidden_dim]
#         attn_output = attn_output.view(total_seq_len, hidden_dim)

#         # Final projection.
#         output = self.Wo(attn_output)

#         # Apply dropout
#         output = self.dropout(output)
#         return output


# class FlashCausalAttentionBlock(nn.Module):
#     """
#     Attention block implementing multi-head causal (masked) attention using
#     Flash Attention.
#     """

#     def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
#         super().__init__()
#         if not FLASH_ATTN_AVAILABLE:
#             raise ImportError("Flash Attention is not available. Install flash-attn package or choose another approach.")

#         assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.head_dim = hidden_dim // num_heads

#         # Projection layers.
#         self.Wq = nn.Linear(hidden_dim, hidden_dim)
#         self.Wk = nn.Linear(hidden_dim, hidden_dim)
#         self.Wv = nn.Linear(hidden_dim, hidden_dim)

#         # Output projection.
#         self.Wo = nn.Linear(hidden_dim, hidden_dim)

#         # Optional dropout.
#         self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

#     def forward(self, x: Tensor, cu_seqlens: Tensor, max_seqlen: int) -> Tensor:
#         """
#         Forward pass.

#         Args:
#             x: Input tensor of shape [total_seq_len, hidden_dim].
#             cu_seqlens: Cumulative sequence lengths tensor of shape [batch_size + 1]
#                         Used for variable sequence length batching in Flash Attention
#             max_seqlen: Maximum sequence length in the batch

#         Returns:
#             Tensor of shape [total_seq_len, hidden_dim] after attention.
#         """
#         # Get total sequence length and hidden dimension
#         total_seq_len, hidden_dim = x.shape

#         # Compute Q, K, V projections.
#         q = self.Wq(x)
#         k = self.Wk(x)
#         v = self.Wv(x)

#         # Reshape to separate the heads: [total_seq_len, hidden_dim] -> [total_seq_len, num_heads, head_dim]
#         q = q.view(total_seq_len, self.num_heads, self.head_dim)
#         k = k.view(total_seq_len, self.num_heads, self.head_dim)
#         v = v.view(total_seq_len, self.num_heads, self.head_dim)

#         # Use Flash Attention with variable sequence lengths and causal masking
#         attn_output = flash_attn_varlen_func(
#             q,
#             k,
#             v,
#             cu_seqlens_q=cu_seqlens,
#             cu_seqlens_k=cu_seqlens,
#             max_seqlen_q=max_seqlen,
#             max_seqlen_k=max_seqlen,
#             causal=True,
#         )

#         # Reshape back to original format: [total_seq_len, num_heads, head_dim] -> [total_seq_len, hidden_dim]
#         attn_output = attn_output.view(total_seq_len, hidden_dim)

#         # Final projection.
#         output = self.Wo(attn_output)

#         # Apply dropout
#         output = self.dropout(output)
#         return output
