import torch

def batch_time_mask(x, num_masks=2, mask_width=8):
    """
    Args:
        x: Input tensor (Batch, Seq_Len, Channels)
        num_masks: How many holes to punch
        mask_width: Width of each hole (in 100Hz samples)
    """
    
    B, T, C = x.shape
    # Create a base mask of ones
    mask = torch.ones((B, T), device=x.device, dtype=x.dtype)
    
    for _ in range(num_masks):
        # Generate random start indices for each item in batch
        # shape (B,)
        starts = torch.randint(0, T - mask_width, (B,), device=x.device)
        
        # Vectorized block masking
        # Create a range [0, mask_width-1] and add to starts
        indices = starts.unsqueeze(1) + torch.arange(mask_width, device=x.device)
        
        # Scatter zeros into the mask
        mask.scatter_(1, indices, 0)

    # Apply mask to all channels (B, T, 1) * (B, T, C)
    return x * mask.unsqueeze(-1)