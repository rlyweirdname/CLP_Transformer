from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from model import Seq2SeqCLP
from environment import Item

app = Flask(__name__)
CORS(app) # Cho phép file HTML gọi API mà không bị chặn lỗi CORS

@app.route('/', methods=['GET'])
def home():
    return send_from_directory('.', 'index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# 1. Tải mô hình AI lên bộ nhớ (chỉ tải 1 lần khi bật server)
print("Đang khởi động Server và tải mô hình Transformer...")
model = Seq2SeqCLP()
try:
    model.load_state_dict(torch.load("clp_transformer.pth"))
    model.eval()
    print("✅ Đã nạp mô hình clp_transformer.pth thành công!")
except FileNotFoundError:
    print("❌ LỖI: Không tìm thấy file clp_transformer.pth")

def get_ai_prediction_autoregressive(items):
    """Sử dụng Transformer dự đoán TỪNG BƯỚC MỘT (Autoregressive)"""
    input_seq = [[item.l, item.w, item.h] for item in items]
    src_tensor = torch.tensor([input_seq], dtype=torch.float32)
    
    tgt_seq = [[0.0, 0.0, 0.0]]
    tgt_tensor = torch.tensor([tgt_seq], dtype=torch.float32)
    
    with torch.no_grad():
        for _ in range(len(items)):
            predictions = model(src_tensor, tgt_tensor)
            next_item = predictions[:, -1:, :]
            tgt_tensor = torch.cat([tgt_tensor, next_item], dim=1)
            
    pred_seq = tgt_tensor[0, 1:].cpu().numpy().tolist() # Convert sang list chuẩn để gửi qua Web
    return pred_seq

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    raw_items = data.get('items', [])
    
    # 2. Chuyển đổi dữ liệu JSON từ Web thành Object Item của Python
    items = []
    for it in raw_items:
        items.append(Item(it['id'], it['l'], it['w'], it['h']))
        
    # 3. Đưa cho não AI suy luận
    predicted_seq = get_ai_prediction_autoregressive(items)
    
    # 4. Trả kết quả (thứ tự kích thước) về lại cho Web
    return jsonify({"predicted_sequence": predicted_seq})

if __name__ == '__main__':
    # Chạy server ở port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)
