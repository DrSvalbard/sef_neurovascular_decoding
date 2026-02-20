import torch
import torch.nn as nn
import torch.nn.functional as F

# Relative imports
from .encoder import PCA_Encoder, PCA_CNN_Encoder
from .positional_encoder import PositionalEncoding, LearnablePositionalEncoding
from .transformer import Transformer_
from .decoder import TemporalDecoder, CausalDilatedDecoder

class master_network(nn.Module):
    def __init__(self, num_pcs, embed_dim, n_head, num_layers, max_len=500, output_len = 100):
        '''
        :param self:
        :param num_pcs: Number of principal componants
        :param embed_dim: Size of transformer input
        :param n_head: Number of transformer MHA
        :param num_layers: Number of transformer layers
        '''

        super().__init__()
        self.encoder = PCA_CNN_Encoder(num_pcs, embed_dim, dropout=0.1)
        self.pe = LearnablePositionalEncoding(embed_dim, max_len=max_len)
        self.transformer = Transformer_(embed_dim, n_head, num_layers, dropout=0.1, activation='gelu')
        self.decoder = CausalDilatedDecoder(embed_dim, pred_channel=1)

        self.query_pos = nn.Parameter(torch.randn(1, output_len, embed_dim))
        self.decoder_cross_attn = nn.MultiheadAttention(embed_dim, n_head, batch_first=True, dropout = 0.1)

        self.output_len = output_len

    def forward(self, x):
        # Input [Batch, PCx, Time]
        # Encoder and PE work on last dimension
        x = self.encoder(x)         # -> [Batch, embed_dim, Time]
        # Transformer -> [Batch, Time, embed_dim]
        x = x.transpose(1, 2)       # -> [Batch, Time, embed_dim]
        x = self.pe(x)              # -> [Batch, Time, embed_dim]
        x = self.transformer(x)     # -> [Batch, Time, embed_dim]
        B = x.shape[0]
        queries = self.query_pos.expand(B, -1, -1) # -> [Batch, Time, embed_dim]
        # LFP (len_LFP) ask fUS (Time)
        x_decoded, _ = self.decoder_cross_attn(queries, x , x) # -> [Batch, len_LFP, 64]
        # Decoder is 1D, need [Batch, embed_dim, Time]
        x_decoded = x_decoded.transpose(2, 1).contiguous()  # -> [Batch, embed_dim, len_LFP]
        x_decoded = self.decoder(x_decoded)         # -> [Batch, Chan_LFP, len_LFP]

        # Soft smoothing
        out = F.avg_pool1d(x_decoded, kernel_size=3, stride=1, padding=1)

        return out
    