import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import Seq2SeqCLP

# 1. ĐỊNH NGHĨA LỚP ĐỌC DỮ LIỆU
class CLPDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Chuyển đổi list (l, w, h) thành PyTorch Tensor kiểu float
        input_seq = torch.tensor(self.data[idx]["input"], dtype=torch.float32)
        target_seq = torch.tensor(self.data[idx]["target"], dtype=torch.float32)
        return input_seq, target_seq

def train_model():
    # 2. CÀI ĐẶT THÔNG SỐ HUẤN LUYỆN (Hyperparameters)
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10 # Số vòng lặp qua toàn bộ dữ liệu
    
    # Load dữ liệu từ file bạn vừa tạo
    print("Đang tải dữ liệu...")
    dataset = CLPDataset("clp_dataset_optimal.json")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. KHỞI TẠO MÔ HÌNH, HÀM MẤT MÁT VÀ BỘ TỐI ƯU
    model = Seq2SeqCLP(n_layers=3)
    criterion = nn.MSELoss() # Dùng MSE vì dự đoán tọa độ/kích thước là bài toán Hồi quy (Regression)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4. VÒNG LẶP HUẤN LUYỆN
    print("Bắt đầu huấn luyện...")
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            # Xóa gradient cũ
            optimizer.zero_grad()
            
            # Đưa dữ liệu qua mô hình
            # Kỹ thuật "Teacher Forcing": Đưa chính Target vào Decoder để dự đoán chính Target
            output = model(src, tgt)
            
            # Tính sai số
            loss = criterion(output, tgt)
            
            # Lan truyền ngược (Backpropagation) và cập nhật trọng số
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # In ra tiến độ học sau mỗi epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Sai số (Loss): {avg_loss:.4f}")
        
    # Lưu lại mô hình sau khi học xong
    torch.save(model.state_dict(), "clp_transformer.pth")
    print("Đã huấn luyện xong và lưu mô hình vào file 'clp_transformer.pth'!")

if __name__ == "__main__":
    train_model()