import torch
import torch.nn as nn

class Seq2SeqCLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, n_heads=4, n_layers=2):
        """
        Mô hình Transformer Seq2Seq.
        - input_dim: 3 (L, W, H của kiện hàng)
        - Đầu ra dự đoán trực tiếp (L, W, H) của kiện hàng nên được xếp tiếp theo.
        """
        super(Seq2SeqCLP, self).__init__()
        
        # Nhúng (Embedding) kích thước kiện hàng thành vector đặc trưng
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Lõi Transformer với cả Encoder và Decoder
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            batch_first=True
        )
        
        # Lớp xuất dữ liệu (dự đoán lại 3 giá trị L, W, H)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, src, tgt):
        """
        src: Tensor chứa danh sách kiện hàng đầu vào (Input)
        tgt: Tensor chứa danh sách kiện hàng mục tiêu (Target) dùng để mớm cho Decoder
        """
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        
        # Đưa qua mạng Transformer
        out = self.transformer(src_emb, tgt_emb)
        
        # Dự đoán kích thước kiện hàng
        predictions = self.fc_out(out)
        return predictions