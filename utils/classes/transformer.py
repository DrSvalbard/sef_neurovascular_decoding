import torch
import torch.nn as nn

class Transformer_(nn.Module):
    def __init__(self, embed_dim, n_head, num_layers, dropout=0.2, activation='gelu' ,attn_dropout=0.2):
        '''
        :param self: 
        :param embed_dim: int, size : output of encoder
        :param n_head: number of MHA (usually 4)
        :param num_layers: number of transformer layer (start with 4)
        :param dropout: dropout
        :param activation: activation function 'gelu' or 'relu'
        '''
        super().__init__()
        trans_layer = nn.TransformerEncoderLayer(
            d_model= embed_dim,
            nhead= n_head,
            dim_feedforward= embed_dim,
            dropout= dropout,
            batch_first= True,
            norm_first= True,
            activation= activation,
        )

        self.transformer = nn.TransformerEncoder(trans_layer,
                                                 num_layers=num_layers)
        
    def forward(self, x):
        # MHA
        x = self.transformer(x)
        # Permute back to [Batch, PCx, Time]
        return x