import torch
import torch.nn as nn
import copy


def make_seq_mask(x, pad_idx):
    sz = x.shape[len(x.shape) == 2]
    mask = (torch.triu(torch.ones((sz, sz), device='cuda')) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    padding_mask = (x == pad_idx)
    return mask, padding_mask


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, ff_dim, dropout, batch_first=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.att_lnorm = nn.LayerNorm(d_model)
        self.ff_lnorm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask, pad_mask):
        x_n = self.att_lnorm(x)
        x_n, _ = self.attention(x, x, x, attn_mask=attn_mask, key_padding_mask=pad_mask)
        x_n = x + self.dropout(x_n)
        x_n = x_n + self.ff(self.ff_lnorm(x_n))
        return x_n


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
    
    def forward(self, x, attn_mask, pad_mask):
        for layer in self.decoder:
            x = layer(x, attn_mask, pad_mask)
        return x
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos_enc = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).reshape(-1, 1)
        den = torch.exp(-torch.arange(0, d_model, 2) * torch.log(torch.Tensor([10000])).item() / d_model)
        # denom = torch.pow(10000, (torch.arange(self.d_model) - (torch.arange(self.d_model) % 2)) / d_model)
        # pe = pos / den
        pos_enc[:, 0::2] = torch.sin(pos / den)
        pos_enc[:, 1::2] = torch.cos(pos / den)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pos_enc', pos_enc, persistent=False)

    def forward(self, x):
        return x + self.pos_enc[:, :x.shape[-2], :]


class LLAMA(nn.Module):
    def __init__(self, num_layers, d_model, nhead, vocab_size, ff_dim, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.decoder = Decoder(
            DecoderLayer(d_model, nhead, ff_dim, dropout),
            num_layers=num_layers
        )
        self.linear = nn.Linear(d_model, vocab_size)

        print(sum([p.numel() for p in self.parameters() if p.requires_grad]), 'parameters :)')
    
    def forward(self, input_ids, attn_mask, pad_mask):
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.decoder(x, attn_mask, pad_mask)
        return self.linear(x)
    
    def get_next_token(self, prefix, mask, pad_mask):
        return self.forward(prefix, mask, pad_mask)[:, -1, :]
    