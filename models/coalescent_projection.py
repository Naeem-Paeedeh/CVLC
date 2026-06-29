import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import einops as eo
from torch import Tensor as T
from typing import Optional, Tuple
import new_types as nt


def initialize_a_learnable_tensor(
    dimensions: list | tuple,
    initialization: nt.InitializationType,
    device,
    dtype,
    std: float = 0.02
):
    var = torch.zeros(*dimensions, dtype=dtype, device=device)
            
    if initialization == nt.InitializationType.Normal:
        var = trunc_normal_(var, std=std)
    elif initialization == nt.InitializationType.Kaiming:
        var = nn.init.kaiming_normal_(var)
    elif initialization == nt.InitializationType.Uniform:
        var = nn.init.uniform_(var, -0.08, 0.08)
    elif initialization in [nt.InitializationType.CloseToDiagonalMatrix, nt.InitializationType.Diagonal]:
        if initialization == nt.InitializationType.CloseToDiagonalMatrix:
            var = trunc_normal_(var, std=std)
        
        if len(dimensions) == 2:
            var.fill_diagonal_(1.0)
        elif len(dimensions) == 3:
            for head_id in range(dimensions[0]):
                var[head_id].fill_diagonal_(1)
    else:
        raise NotImplementedError
        
    var = nn.Parameter(var)
        
    return var


def initialize_a_coalescent_projection_tensor(
    num_heads: int,
    dim_head: int,
    shared_across_heads: bool,
    device=None,
    dtype=torch.float32,
    std: float = 0.02,
    initialization: nt.InitializationType = nt.InitializationType.CloseToDiagonalMatrix,
):
    # dim_embed = num_heads * dim_head
    
    if not shared_across_heads:     # Default
        dimensions = [num_heads, dim_head, dim_head]
    else:
        dimensions = [dim_head, dim_head]
    cp = initialize_a_learnable_tensor(dimensions, std=std, initialization=initialization, dtype=dtype, device=device)
    
    return cp


# Modified from the PyTorch's implementation.
def MultiheadAttention_manual_forward(
    mha: nn.MultiheadAttention,
    query: T,
    key: T,
    value: T,
    CPs_QK: Optional[T] = None,
    CPs_SV: Optional[T] = None, # Added Coalesced Value Projection
    lora_qkv = None,
    attn_mask: Optional[T] = None,
    key_padding_mask: Optional[T] = None,
    need_weights: bool = True,
    average_attn_weights: bool = True,
    training: bool = False  # Explicitly control training mode for dropout
) -> Tuple[T, Optional[T]]:
    # 1. Extract Parameters and Check Configuration
    dim_embed = mha.embed_dim
    num_heads = mha.num_heads
    dropout = mha.dropout
    dim_head = mha.head_dim
    batch_first = mha.batch_first
    bias = mha.in_proj_bias is not None or mha.bias_k is not None  # Check if any bias exists

    kdim = mha.kdim
    vdim = mha.vdim
    assert kdim == key.size(-1), f"key's feature dimension ({key.size(-1)}) must match kdim ({kdim})"
    assert vdim == value.size(-1), f"value's feature dimension ({value.size(-1)}) must match vdim ({vdim})"
    assert query.size(-1) == dim_embed, f"query's feature dimension ({query.size(-1)}) must match dim_embed ({dim_embed})"

    # 2. Handle batch_first
    # Internally, we'll mostly work with (SeqLen, Batch, Dim) format for simplicity
    if batch_first:
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

    target_length, batch_size, _ = query.shape
    source_length, _, _ = key.shape
    assert key.size(1) == batch_size, f"key's batch size ({key.size(1)}) must match query's batch size ({batch_size})"
    assert value.size(1) == batch_size, f"value's batch size ({value.size(1)}) must match query's batch size ({batch_size})"
    assert value.size(0) == source_length, f"value's sequence length ({value.size(0)}) must match key's sequence length ({source_length})"

    # 3. Input Projections
    # Check if using combined projection weight or separate ones
    if mha.in_proj_weight is not None:
        # Combined weights: W_q, W_k, W_v are concatenated along dim 0
        # Bias: b_q, b_k, b_v are concatenated along dim 0
        assert kdim == dim_embed, "kdim must equal dim_embed for combined projection"
        assert vdim == dim_embed, "vdim must equal dim_embed for combined projection"

        q_proj_weight = mha.in_proj_weight[:dim_embed, :]
        k_proj_weight = mha.in_proj_weight[dim_embed:2 * dim_embed, :]
        v_proj_weight = mha.in_proj_weight[2 * dim_embed:, :]

        q_proj_bias = None
        k_proj_bias = None
        v_proj_bias = None
        if bias and mha.in_proj_bias is not None:
            q_proj_bias = mha.in_proj_bias[:dim_embed]
            k_proj_bias = mha.in_proj_bias[dim_embed:2 * dim_embed]
            v_proj_bias = mha.in_proj_bias[2 * dim_embed:]

        # Apply linear projection: output = input @ weight.T + bias
        q = F.linear(query, q_proj_weight, q_proj_bias)
        k = F.linear(key, k_proj_weight, k_proj_bias)
        v = F.linear(value, v_proj_weight, v_proj_bias)

    else:
        # Separate weights and potentially biases
        q = F.linear(query, mha.q_proj_weight, mha.in_proj_bias)  # Bias for Q is stored in in_proj_bias
        k = F.linear(key, mha.k_proj_weight, mha.bias_k)
        v = F.linear(value, mha.v_proj_weight, mha.bias_v)
    
    # LoRAs
    if lora_qkv is not None:
        if lora_qkv.lora_qkv_mask[0]:
            q = q + lora_qkv.adapters_list[0].forward(query)
        if lora_qkv.lora_qkv_mask[1]:
            k = k + lora_qkv.adapters_list[1].forward(key)
        if lora_qkv.lora_qkv_mask[2]:
            v = v + lora_qkv.adapters_list[2].forward(value)
    
    # Coalescent Projections
    if CPs_QK is not None:
        # q.shape: (SeqLen, Batch, EmbedDim)
        q_processed = eo.rearrange(q, "s b (nh dh) -> s nh b dh", nh=num_heads)
        q_processed = q_processed @ CPs_QK
        q = eo.rearrange(q_processed, "s nh b dh -> s b (nh dh)")

    # Coalesced Value Projection
    if CPs_SV is not None:
        # v.shape: (SeqLen, Batch, EmbedDim)
        v_processed = eo.rearrange(v, "s b (nh dh) -> s nh b dh", nh=num_heads)
        v_processed = v_processed @ CPs_SV
        v = eo.rearrange(v_processed, "s nh b dh -> s b (nh dh)")

    # 4. Reshape for Multihead Processing
    # Reshape q, k, v from (SeqLen, Batch, EmbedDim) to (SeqLen, Batch * NumHeads, HeadDim)
    # Then transpose to (Batch * NumHeads, SeqLen, HeadDim) for batch matrix multiplication
    q = q.contiguous().view(target_length, batch_size * num_heads, dim_head).transpose(0, 1)
    k = k.contiguous().view(source_length, batch_size * num_heads, dim_head).transpose(0, 1)
    v = v.contiguous().view(source_length, batch_size * num_heads, dim_head).transpose(0, 1)
    # Now q: (N * num_heads, L, head_dim), k: (N * num_heads, S, head_dim), v: (N * num_heads, S, dim_head)

    # 5. Prepare Masks
    # attn_mask: (L, S) or (N * num_heads, L, S)
    # key_padding_mask: (N, S)
    merged_mask = None
    if attn_mask is not None:
        # Ensure attn_mask is broadcastable to (N * num_heads, L, S)
        if attn_mask.dim() == 2:
            # Shape (L, S) -> (1, L, S)
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            # Shape (N * num_heads, L, S) - already correct
            pass
        else:
            raise ValueError(f"attn_mask expected 2D or 3D, got {attn_mask.dim()}D")

        # Ensure boolean type or float type
        if attn_mask.dtype == torch.bool:
            # Convert boolean mask to float mask (-inf for True positions)
            attn_mask = torch.where(attn_mask, float('-inf'), float(0.0)).to(q.dtype)
        elif attn_mask.dtype != q.dtype:
            attn_mask = attn_mask.to(q.dtype)

        # Check broadcasting compatibility (N dimension)
        if attn_mask.size(0) == 1:
            # Expand N dimension if needed (from 1 to N * num_heads)
            attn_mask = attn_mask.expand(batch_size * num_heads, -1, -1)
        elif attn_mask.size(0) != batch_size * num_heads:
            raise ValueError(f"attn_mask batch dimension ({attn_mask.size(0)}) must be 1 or equal to batch_size * num_heads ({batch_size * num_heads})")

        merged_mask = attn_mask  # Shape: (N * num_heads, L, S)

    if key_padding_mask is not None:
        # Shape (N, S) -> (N, 1, 1, S) -> (N * num_heads, 1, S) or broadcast
        assert key_padding_mask.dim() == 2, f"key_padding_mask expected 2D, got {key_padding_mask.dim()}D"
        assert key_padding_mask.shape == (batch_size, source_length), \
            f"key_padding_mask shape expected ({batch_size}, {source_length}), got {key_padding_mask.shape}"

        # Convert boolean mask to float mask (-inf for True positions)
        if key_padding_mask.dtype == torch.bool:
            float_kp_mask = torch.where(key_padding_mask, float('-inf'), float(0.0)).to(q.dtype)
        else:   # Assume it's already a float mask
            float_kp_mask = key_padding_mask.to(q.dtype)

        # Reshape and expand for broadcasting: (N, S) -> (N, 1, 1, S) -> (N, num_heads, 1, S) -> (N*num_heads, 1, S)
        float_kp_mask = float_kp_mask.view(batch_size, 1, 1, source_length)
        float_kp_mask = float_kp_mask.expand(-1, num_heads, -1, -1)         # Expand along head dim
        float_kp_mask = float_kp_mask.reshape(batch_size * num_heads, 1, source_length)  # Reshape to (N*num_heads, 1, S)

        # Add to the existing mask or create if none exists
        if merged_mask is None:
            merged_mask = float_kp_mask  # Broadcasts L dim: (N*num_heads, 1, S) -> (N*num_heads, L, S)
        else:
            # Add masks together. Broadcasting handles the dimensions.
            # merged_mask: (N*num_heads, L, S)
            # float_kp_mask: (N*num_heads, 1, S) -> broadcasts to (N*num_heads, L, S)
            merged_mask = merged_mask + float_kp_mask

    # 6. Scaled Dot-Product Attention
    # Scaling factor
    scaling_factor = float(dim_head) ** -0.5
    q = q * scaling_factor

    k_transpose = k.transpose(1, 2)     # (batch_size * num_heads, dim_head, SeqLen)
    
    attn_output_weights = q @ k_transpose

    # Apply the merged mask (attn_mask + key_padding_mask)
    if merged_mask is not None:
        # Ensure mask shape matches attn_output_weights shape
        if merged_mask.shape != attn_output_weights.shape:
            # This might happen if broadcasting didn't work as expected, raise error
            raise RuntimeError(f"Mask shape {merged_mask.shape} does not match attention score shape {attn_output_weights.shape}")
        attn_output_weights += merged_mask

    # Apply softmax
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)

    # Apply dropout (only during training)
    if dropout > 0.0 and training:
        attn_output_weights = F.dropout(attn_output_weights, p=dropout, training=training)

    # Calculate weighted sum of values: (N * num_heads, L, S) @ (N * num_heads, S, dim_head) -> (N * num_heads, L, dim_head)
    attn_output = torch.bmm(attn_output_weights, v)

    # 7. Reshape and Concatenate Heads
    # attn_output: (N * num_heads, L, dim_head) -> (L, N * num_heads, dim_head) -> (L, N, num_heads * dim_head) = (L, N, E)
    attn_output = attn_output.transpose(0, 1).contiguous().view(target_length, batch_size, dim_embed)

    # 8. Output Projection
    attn_output = F.linear(attn_output, mha.out_proj.weight, mha.out_proj.bias)

    # 9. Handle batch_first for output
    if batch_first:
        attn_output = attn_output.transpose(0, 1)  # (N, L, E)

    # 10. Prepare Attention Weights Output
    final_attn_weights = None
    if need_weights:
        # attn_output_weights shape is (N * num_heads, L, S)
        # Reshape to (N, num_heads, L, S)
        attn_output_weights = attn_output_weights.view(batch_size, num_heads, target_length, source_length)
        if average_attn_weights:
            # Average across heads -> (N, L, S)
            final_attn_weights = attn_output_weights.mean(dim=1)
        else:
            # Keep per-head weights -> (N, num_heads, L, S)
            final_attn_weights = attn_output_weights

    return attn_output, final_attn_weights


# Reference:
# https://github.com/Naeem-Paeedeh/CPLSR
