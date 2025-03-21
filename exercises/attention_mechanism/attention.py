import torch
import torch.nn.functional as F
from torch import BoolTensor, Tensor

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


def eager_bidirectional_attention(
    q: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    k: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    v: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    mask: BoolTensor | None = None,  # shape: [batch_size, sequence_length] where True indicates attended positions
) -> Tensor:
    """
    Implement bidirectional (full) attention using only PyTorch operations.

    Args:
        q: Query tensor of shape [batch_size, sequence_length, hidden_dim]
        k: Key tensor of shape [batch_size, sequence_length, hidden_dim]
        v: Value tensor of shape [batch_size, sequence_length, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        mask: Optional boolean mask where True indicates attended tokens and
            False masked positions. Can be of shape either [batch_size,
            sequence_length] indicating which tokens other tokens should attend
            to or [batch_size, sequence_length, sequence_length] indicating
            which tokens (dim 1) should attend to which other tokens (dim 2).

    Returns:
        Output tensor of shape [batch_size, sequence_length, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.

    Note:
        You need to reshape the inputs to separate the heads, perform the
        attention computation, and then merge the heads back. You might need
        to invert the attention mask for the softmax to attend correctly.
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
        attn_scores.masked_fill_(~mask[:, None, None] if mask.dim() == 2 else ~mask[:, None], -torch.inf)
    attn_weights = torch.softmax(attn_scores / head_dim**0.5, dim=-1)

    return consolidate_over_heads(attn_weights @ v, num_heads=num_heads)


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
        )[None]
        .bool()
        .to(_get_device())
    )


def eager_causal_attention(
    q: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    k: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    v: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    mask: BoolTensor | None = None,  # shape: [batch_size, sequence_length] where True indicates attended positions
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
              where True indicates attended tokens and False masked positions

    Returns:
        Output tensor of shape [batch_size, sequence_length, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.

    Note:
        A causal mask ensures that a position i can only attend to positions j ≤ i.
        You might need to invert the attention mask for the softmax to attend correctly.
    """
    sequence_length = q.shape[1]

    causal_mask = generate_causal_mask(sequence_length)

    return eager_bidirectional_attention(
        q=q,
        k=k,
        v=v,
        num_heads=num_heads,
        head_dim=head_dim,
        mask=causal_mask if mask is None else causal_mask & mask[:, None],
    )


def sdpa_bidirectional_attention(
    q: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    k: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    v: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    mask: BoolTensor | None = None,  # shape: [batch_size, sequence_length] where True indicates attended positions
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
              where True indicates attended tokens and False masked positions

    Returns:
        Output tensor of shape [batch_size, sequence_length, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.
    """
    return consolidate_over_heads(
        F.scaled_dot_product_attention(
            query=distribute_over_heads(q, num_heads=num_heads),
            key=distribute_over_heads(k, num_heads=num_heads),
            value=distribute_over_heads(v, num_heads=num_heads),
            attn_mask=mask if mask is None else mask[:, None, None],
        ),
        num_heads=num_heads,
    )


def sdpa_causal_attention(
    q: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    k: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    v: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    mask: BoolTensor | None = None,  # shape: [batch_size, sequence_length] where True indicates attended positions
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
              where True indicates attended tokens and False masked positions

    Returns:
        Output tensor of shape [batch_size, sequence_length, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.

    Note:
        You can use the `is_causal` argument to enable causal masking instead of
        creating a causal mask.
    """
    sequence_length = q.shape[1]
    causal_mask = generate_causal_mask(sequence_length)
    return consolidate_over_heads(
        F.scaled_dot_product_attention(
            query=distribute_over_heads(q, num_heads=num_heads),
            key=distribute_over_heads(k, num_heads=num_heads),
            value=distribute_over_heads(v, num_heads=num_heads),
            attn_mask=causal_mask if mask is None else causal_mask & mask[:, None, None],
        ),
        num_heads=num_heads,
    )


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
    total_seq_len = q.shape[0]

    return flash_attn_varlen_func(
        q=q.view(total_seq_len, num_heads, head_dim),
        k=k.view(total_seq_len, num_heads, head_dim),
        v=v.view(total_seq_len, num_heads, head_dim),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
    ).reshape(q.shape)


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
    total_seq_len = q.shape[0]

    return flash_attn_varlen_func(
        q=q.view(total_seq_len, num_heads, head_dim),
        k=k.view(total_seq_len, num_heads, head_dim),
        v=v.view(total_seq_len, num_heads, head_dim),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    ).reshape(q.shape)
