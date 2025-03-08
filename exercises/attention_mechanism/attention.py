from __future__ import annotations

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


def eager_bidirectional_attention(
    q: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    k: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    v: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    mask: BoolTensor | None = None,  # shape: [batch_size, sequence_length] where True indicates masked positions
) -> Tensor:
    """
    Implement bidirectional (full) attention using only PyTorch operations.

    Args:
        q: Query tensor of shape [batch_size, sequence_length, hidden_dim]
        k: Key tensor of shape [batch_size, sequence_length, hidden_dim]
        v: Value tensor of shape [batch_size, sequence_length, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        mask: Optional boolean mask where True indicates masked positions.
            Can be of shape either [batch_size, sequence_length] indicating
            which tokens that no tokens should attend to or
            [batch_size, sequence_length, sequence_length] indicating which
            tokens (dim 1) should not attend to which other tokens (dim 2).

    Returns:
        Output tensor of shape [batch_size, sequence_length, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.

    Note:
        You need to reshape the inputs to separate the heads, perform the
        attention computation, and then merge the heads back.
    """
    k = distribute_over_heads(k, num_heads=num_heads)
    q = distribute_over_heads(q, num_heads=num_heads)
    v = distribute_over_heads(v, num_heads=num_heads)

    # Matrix multiply with head_dim as inner dimension leaves
    # batch_size x num_heads x num_tokens x num_tokens
    attn_scores = q @ k.transpose(-2, -1)
    if mask is not None:
        # repeat mask along num_heads and the first num_tokens dimension
        # so that each token ignores the specified tokens
        attn_scores.masked_fill_(
            mask[:, None, None] if mask.dim() == 2 else mask[:, None], -torch.inf
        )
    attn_weights = torch.softmax(attn_scores / head_dim**0.5, dim=-1)

    return consolidate_over_heads(attn_weights @ v, num_heads=num_heads)


def eager_causal_attention(
    q: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    k: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    v: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    mask: BoolTensor | None = None,  # shape: [batch_size, sequence_length] where True indicates masked positions
) -> Tensor:
    """
    Implement causal (masked) attention using only PyTorch operations.

    Args:
        q: Query tensor of shape [batch_size, sequence_length, hidden_dim]
        k: Key tensor of shape [batch_size, sequence_length, hidden_dim]
        v: Value tensor of shape [batch_size, sequence_length, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        mask: Optional boolean mask of shape [batch_size, sequence_length]
              where True indicates masked positions

    Returns:
        Output tensor of shape [batch_size, sequence_length, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.

    Note:
        A causal mask ensures that a position i can only attend to positions j ≤ i.
    """
    batch_size, num_tokens, _ = k.shape
    causal_mask = (
        torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1)
        .to(q.device)
        .bool()
        .expand(batch_size, -1, -1)
    )
    return eager_bidirectional_attention(
        k=k,
        q=q,
        v=v,
        num_heads=num_heads,
        head_dim=head_dim,
        mask=causal_mask if mask is None else torch.logical_or(mask[:, None], causal_mask)
    )


def sdp_bidirectional_attention(
    q: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    k: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    v: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    mask: BoolTensor | None = None,  # shape: [batch_size, sequence_length] where True indicates masked positions
) -> Tensor:
    """
    Implement bidirectional (full) attention using PyTorch's scaled_dot_product_attention.

    Args:
        q: Query tensor of shape [batch_size, sequence_length, hidden_dim]
        k: Key tensor of shape [batch_size, sequence_length, hidden_dim]
        v: Value tensor of shape [batch_size, sequence_length, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        mask: Optional boolean mask of shape [batch_size, sequence_length]
              where True indicates masked positions

    Returns:
        Output tensor of shape [batch_size, sequence_length, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.

    Note:
        Note that there's a difference in mask interpretation between our interface and
        PyTorch's SDPA function. In our interface, True means "masked out", while in
        PyTorch's SDPA, True means "participate in attention".
    """
    return consolidate_over_heads(
        F.scaled_dot_product_attention(
            query=distribute_over_heads(q, num_heads=num_heads),
            key=distribute_over_heads(k, num_heads=num_heads),
            value=distribute_over_heads(v, num_heads=num_heads),
            attn_mask=~mask[:, None, None],
        ),
        num_heads=num_heads,
    )


def sdp_causal_attention(
    q: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    k: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    v: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    mask: BoolTensor | None = None,  # shape: [batch_size, sequence_length] where True indicates masked positions
) -> Tensor:
    """
    Implement causal (masked) attention using PyTorch's scaled_dot_product_attention.

    Args:
        q: Query tensor of shape [batch_size, sequence_length, hidden_dim]
        k: Key tensor of shape [batch_size, sequence_length, hidden_dim]
        v: Value tensor of shape [batch_size, sequence_length, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        mask: Optional boolean mask of shape [batch_size, sequence_length]
              where True indicates masked positions

    Returns:
        Output tensor of shape [batch_size, sequence_length, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.

    Note:
        Note that there's a difference in mask interpretation between our interface and
        PyTorch's SDPA function. In our interface, True means "masked out", while in
        PyTorch's SDPA, True means "participate in attention".

        You can use the `is_causal` argument to enable causal masking instead of
        creating a causal mask.
    """
    raise NotImplementedError("Implement causal attention using PyTorch's SDPA")


def flash_bidirectional_attention(
    q: Tensor,  # shape: [total_seq_len, hidden_dim]
    k: Tensor,  # shape: [total_seq_len, hidden_dim]
    v: Tensor,  # shape: [total_seq_len, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    cu_seqlens: Tensor,  # shape: [batch_size + 1], cumulative sequence lengths
    max_seqlen: int,  # maximum sequence length
) -> Tensor:
    """
    Implement bidirectional (full) attention using Flash Attention.

    Args:
        q: Query tensor of shape [total_seq_len, hidden_dim]
        k: Key tensor of shape [total_seq_len, hidden_dim]
        v: Value tensor of shape [total_seq_len, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
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
        Output tensor of shape [total_seq_len, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.

    Note:
        Flash Attention uses a different interface than scaled_dot_product_attention.
        Instead of using an attention mask, it uses cumulative sequence lengths (cu_seqlens)
        and the maximum sequence length (max_seqlen) to .
    """
    raise NotImplementedError("Implement bidirectional attention using Flash Attention")


def flash_causal_attention(
    q: Tensor,  # shape: [total_seq_len, hidden_dim]
    k: Tensor,  # shape: [total_seq_len, hidden_dim]
    v: Tensor,  # shape: [total_seq_len, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    cu_seqlens: Tensor,  # shape: [batch_size + 1], cumulative sequence lengths
    max_seqlen: int,  # maximum sequence length
) -> Tensor:
    """
    Implement causal (masked) attention using Flash Attention.

    Args:
        q: Query tensor of shape [total_seq_len, hidden_dim]
        k: Key tensor of shape [total_seq_len, hidden_dim]
        v: Value tensor of shape [total_seq_len, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
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
        Output tensor of shape [total_seq_len, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.

    Note:
        Flash Attention uses a different interface than scaled_dot_product_attention.
        Instead of using masks, it uses cumulative sequence lengths (cu_seqlens)
        and the maximum sequence length (max_seqlen) to handle variable-length sequences.

        For causal attention, you'll need to set the causal flag to True when using
        the Flash Attention function.
    """
    raise NotImplementedError("Implement causal attention using Flash Attention")
