import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalDecoder(nn.Module):
    def __init__(self, embed_dim, pred_channel=1, output_len=120):
        '''
        :param self:
        :param embed_dim: Entry (Transformer output)
        :param pred_channel: int, Number of LFP predicted 
        :param output_len: Len of predicted LFP (1=10ms)
        '''
        super().__init__()
        self.output_len = output_len

        self.conv_block = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=15, padding=7), # padding is krnl//2
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 64, kernel_size=7, stride=2 ,padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, pred_channel, kernel_size=5, padding=2) # pred_channel: 1 for only one LFP chan, more if needed
        )

    def forward(self, x):

        x = self.conv_block(x)
        
        if x.shape[-1] != self.output_len:
            x = F.interpolate(x, size=self.output_len, mode='linear', align_corners=False)


        return x
    

class CausalDilatedDecoder(nn.Module):
    def __init__(self, embed_dim=64, pred_channel=1):
        super().__init__()
        # Dilation 1 & 2 expands receptive field without losing resolution
        self.conv1 = nn.Conv1d(embed_dim, 32, kernel_size=5, padding=0, dilation=1)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=5, padding=0, dilation=2)
        self.conv3 = nn.Conv1d(16, pred_channel, kernel_size=3)

    def forward(self, x):
        # Pad (kernel-1)*dilation to maintain length 100
        x = F.pad(x, (4, 0)) # for conv1
        x = F.gelu(self.conv1(x))
        
        x = F.pad(x, (8, 0)) # for conv2
        x = F.gelu(self.conv2(x))

        x = F.pad(x, (2, 0)) #conv 3
        return self.conv3(x) # Output: [Batch, 1, 100]