import torch
import torch.nn as nn 

class PCA_Encoder(nn.Module):
    '''
    INIT
    '''
    def __init__(self, num_pcs, embed_dim, dropout=0.1):
        '''
        Docstring pour __init__
        :param self: 
        :param num_pcs: Number of kept principal componant from the PCA
        :param embed_dim: Output (Transformer input)
        :param dropout: Dropout layer
        '''
        super().__init__()
        self.projection = nn.Linear(num_pcs, embed_dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim) # Transformer stabilization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.projection(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class PCA_CNN_Encoder(nn.Module):
    def __init__(self, num_pcs, embed_dim, padding1=30, padding2=10, dropout=0.1):
        '''
        :param self:
        :param num_pcs: int, number of kept PC from the PCA
        :param embed_dim: Output (Transformer Layout)
        :param dropout: Dropout
        '''
        super().__init__()

        self.padding1 = padding1
        self.padding2 = padding2

        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(num_pcs, 32, kernel_size=padding1+1, padding=0), # -> 0 because causal padding
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout1d(dropout))
        
        self.cnn_encoder2 = nn.Sequential(
            nn.Conv1d(32, embed_dim, kernel_size=padding2+1, padding=0), # -> Projection to the transformer layer
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )

        self.shortcut = nn.Sequential(
            nn.Conv1d(num_pcs, embed_dim, kernel_size=1, padding=0),
            nn.BatchNorm1d(embed_dim)
        )

        self.end_gelu = nn.GELU()

    def forward(self, x):
        # Causal padding
        x_padded = nn.functional.pad(x, (self.padding1, 0))
        # Conv1
        conv_out = self.cnn_encoder(x_padded)
        # repad causal
        conv_out = nn.functional.pad(conv_out, (self.padding2, 0))
        # Reconv
        conv_out = self.cnn_encoder2(conv_out)
        # Skip
        res = self.shortcut(x)
        # Skip + Conv add
        out = conv_out + res

        return self.end_gelu(out)