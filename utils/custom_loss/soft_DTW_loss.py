import torch.nn as nn
import torch.nn.functional as F
import torch

class LFPCompetitionLoss(nn.Module):
    def __init__(self, gamma=0.1, alpha=0.3, eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha # Weight for DTW
        self.eps = eps

    def soft_dtw(self, pred, target):
        # B, N (Sequence Length)
        B, N = pred.shape
        # Euclidean distance matrix (B, N, N)
        dist_mat = torch.cdist(pred.unsqueeze(-1), target.unsqueeze(-1), p=2)**2
        
        # DP Initialization
        R = torch.full((B, N + 1, N + 1), 1e8, device=pred.device)
        R[:, 0, 0] = 0
        
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                # Min(left, top, diag) using logsumexp for differentiability
                v = torch.stack([R[:, i-1, j], R[:, i, j-1], R[:, i-1, j-1]], dim=1)
                soft_min = -self.gamma * torch.logsumexp(-v / self.gamma, dim=1)
                R[:, i, j] = dist_mat[:, i-1, j-1] + soft_min
        
        return R[:, N, N].mean()
    
    def _ensure_2d(self, x):
        if x.ndim == 1:
            # Cas (120,) -> (1, 120)
            return x.unsqueeze(0)
        elif x.ndim == 2:
            # Cas (B, 120) -> Reste (B, 120)
            return x
        elif x.ndim == 3:
            # Cas (B, C, T) -> (B*C, T) pour traiter chaque canal
            B, C, T = x.shape
            return x.reshape(B * C, T)
        return x

    def forward(self, pred, target):
        # pearson = (x - mx)(y - my) / (||x - mx|| * ||y - my||)
        pred = self._ensure_2d(pred)
        target = self._ensure_2d(target)

        p_mu = pred.mean(dim=1, keepdim=True)
        t_mu = target.mean(dim=1, keepdim=True)
        
        p_centered = pred - p_mu
        t_centered = target - t_mu
        
        cos_sim = F.cosine_similarity(p_centered, t_centered, dim=1, eps=self.eps)
        pearson_loss = 1 - cos_sim.mean()
        
        # Add DTW if using for temporal alignment
        if self.alpha > 0:
            dtw_dist = self.soft_dtw(pred, target)
            return (self.alpha * dtw_dist) + ((1 - self.alpha) * pearson_loss)
        
        return pearson_loss