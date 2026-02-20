import torch
import torch.nn as nn
import torch.nn.functional as F

# class TemporalEnsembleLoss(nn.Module):
#     def __init__(self, w_pearson=1.0, w_grad=0.5, w_smooth=0.2):
#         super().__init__()
#         self.w_pearson = w_pearson
#         self.w_grad = w_grad
#         self.w_smooth = w_smooth

#     def forward(self, pred, target):
#         # Ensure shape [Batch, Time]
#         if pred.ndim == 3: pred = pred.squeeze(1)
#         if target.ndim == 3: target = target.squeeze(1)

#         # 1. Pearson Correlation Loss
#         p_mu = pred.mean(dim=1, keepdim=True)
#         t_mu = target.mean(dim=1, keepdim=True)
#         p_std = pred - p_mu
#         t_std = target - t_mu
        
#         # Add epsilon to avoid div by zero
#         cos_sim = F.cosine_similarity(p_std, t_std, dim=1, eps=1e-6)
#         loss_pearson = 1 - cos_sim.mean()

#         # 2. Gradient (Slope) Loss - Penalizes Jitter
#         # dy/dt difference
#         d_pred = pred[:, 1:] - pred[:, :-1]
#         d_target = target[:, 1:] - target[:, :-1]
#         loss_grad = F.mse_loss(d_pred, d_target)

#         # 3. Multi-Scale Smoothness (Mean Pooling)
#         # Force energy to match at lower frequencies (the 1-2s trend)
#         loss_smooth = 0
#         for k in [5, 11, 31]: # Look at 50ms and 110ms and 310 ms chunks
#             p_avg = F.avg_pool1d(pred.unsqueeze(1), kernel_size=k, stride=1, padding=k//2)
#             t_avg = F.avg_pool1d(target.unsqueeze(1), kernel_size=k, stride=1, padding=k//2)
#             loss_smooth += F.mse_loss(p_avg, t_avg)

#         # Total Weighted Loss
#         total_loss = (self.w_pearson * loss_pearson) + \
#                      (self.w_grad * loss_grad) + \
#                      (self.w_smooth * loss_smooth)
        
#         return total_loss
    

class TemporalEnsembleLoss(nn.Module):
    def __init__(self, w_pearson=10.0, w_grad=0.1, w_sign=0.0, w_var = 5.0 , w_range = 0.0):
        super().__init__()
        self.w_pearson = w_pearson
        self.w_grad = w_grad
        self.w_sign = w_sign
        self.w_var = w_var
        self.w_range = w_range

    def forward(self, pred, target):
        p = pred.squeeze(1) if pred.ndim == 3 else pred
        t = target.squeeze(1) if target.ndim == 3 else target

        # 1. Pearson with safety
        p_mu, t_mu = p.mean(dim=1, keepdim=True), t.mean(dim=1, keepdim=True)
        p_centered, t_centered = p - p_mu, t - t_mu
        
        # Manually compute cosine sim to control epsilon better
        num = (p_centered * t_centered).sum(dim=1)
        den = torch.sqrt((p_centered**2).sum(dim=1) * (t_centered**2).sum(dim=1)) + 1e-8
        loss_pearson = 1 - (num / den).mean()

        # 2. Gradient Loss (Safe)
        d_p, d_t = p[:, 1:] - p[:, :-1], t[:, 1:] - t[:, :-1]
        loss_grad = F.mse_loss(d_p, d_t)

        # 3. Variance Matching (Add eps inside sqrt/std)
        p_std = torch.sqrt(p.var(dim=1) + 1e-8)
        t_std = torch.sqrt(t.var(dim=1) + 1e-8)
        loss_var = F.l1_loss(p_std, t_std)

        # 4. Contrastive Sharpness
        pred_range = p.max(dim=1)[0] - p.min(dim=1)[0]
        target_range = t.max(dim=1)[0] - t.min(dim=1)[0]
        loss_range = F.mse_loss(pred_range, target_range)

        # 5. Direction loss
        sign_match = torch.sign(d_p) * torch.sign(d_t)
        loss_direction = 1 - (sign_match == 1).float().mean()

        return (self.w_pearson * loss_pearson) + (self.w_grad * loss_grad) + \
               (self.w_var * loss_var) + (self.w_range * loss_range) + (self.w_sign * loss_direction)