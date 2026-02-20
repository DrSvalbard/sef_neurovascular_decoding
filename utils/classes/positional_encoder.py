import torch
import torch.nn as nn 
import math

# Vaswani 2017 "Attention is all you need"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=600):
        super().__init__()
        # Matrix [max_len, d_model] (Vaswani et al.)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        
        # register buffer -> MPS w/o model parameter
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # [Batch, Time, Channels]
        x = x + self.pe[:,:x.size(1),:]
        return x
        

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=350):
        super().__init__()
        # Initialize as a learnable parameter [1, max_len, d_model]
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        # x: [Batch, Time, Channels]
        # Adding the learnable bias to each time step
        return x + self.pe[:, :x.size(1), :]