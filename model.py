import torch
import torch.nn as nn
import math
from typing import cast

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # type: Tensor

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        pe = cast(torch.Tensor, self.pe)
        x = x + pe[:, :seq_len, :]
        return x

class Seq2SeqCLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, n_heads=4, n_layers=3, dropout=0.1):
        super(Seq2SeqCLP, self).__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Thêm mã hóa vị trí
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, src, tgt):
        # 1. Nhúng và cộng thêm Mã hóa vị trí
        src_emb = self.pos_encoder(self.embedding(src))
        tgt_emb = self.pos_encoder(self.embedding(tgt))
        
        # FIX: TẠO MẶT NẠ CHỐNG NHÌN TRỘM TƯƠNG LAI (Causal Mask)
        seq_len = tgt.size(1)
        # Tạo ma trận tam giác trên chứa các giá trị -inf để che đi các vị trí j > i
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        
        # 2. Đưa qua mạng Transformer (nhớ truyền thêm tham số tgt_mask)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        
        # 3. Dự đoán kích thước kiện hàng
        predictions = self.fc_out(out)
        return predictions
