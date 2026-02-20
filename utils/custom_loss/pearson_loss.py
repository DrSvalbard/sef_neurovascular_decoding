import torch 

def pearson_loss(output, target):

    x = output.view(output.size(0), -1) # [Batch, Channel ,Time]
    y = target.view(target.size(0), -1) # [Batch, Channel ,Time]


    mx = x - torch.mean(x, dim=-1, keepdim=True)
    my = y - torch.mean(y, dim=-1, keepdim=True)

    sum_mx2 = torch.sum(mx**2, dim=-1)
    sum_my2 = torch.sum(my**2, dim=-1)
    sum_mxy = torch.sum(mx * my, dim=-1)

    norm = torch.sqrt(sum_mx2 * sum_my2 + 1e-9)
    
    corr = torch.clamp(sum_mxy / norm, min=-1.0, max=1.0)
    
    return torch.mean(1 - corr)