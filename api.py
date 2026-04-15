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
    global model, model_hidden_dim, model_error
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_hidden_dim": model_hidden_dim,
        "model_error": model_error
    })

# 1. Tải mô hình AI lên bộ nhớ (chỉ tải 1 lần khi bật server)
print("Đang khởi động Server và tải mô hình Transformer...")
model = None
model_hidden_dim = None
model_error = None

MAX_DIM = 20.0

def normalize(tensor):
    return tensor / MAX_DIM

def denormalize(tensor):
    return tensor * MAX_DIM

def _extract_state_dict(raw_checkpoint):
    if isinstance(raw_checkpoint, dict) and "state_dict" in raw_checkpoint:
        return raw_checkpoint["state_dict"]
    return raw_checkpoint

def _infer_hidden_dim(state_dict, default_dim=128):
    emb_weight = state_dict.get("embedding.weight")
    if emb_weight is not None and hasattr(emb_weight, "shape") and len(emb_weight.shape) == 2:
        return int(emb_weight.shape[0])
    return default_dim

def _infer_n_layers(state_dict, default_n_layers=3):
    layer_indices = set()
    for key in state_dict.keys():
        parts = key.split(".")
        if "layers" in parts:
            idx = parts.index("layers")
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                layer_indices.add(int(parts[idx + 1]))
    return max(layer_indices) + 1 if layer_indices else default_n_layers

try:
    raw_checkpoint = torch.load("clp_transformer.pth", map_location="cpu")
    state_dict = _extract_state_dict(raw_checkpoint)
    model_hidden_dim = _infer_hidden_dim(state_dict, default_dim=128)
    model_n_layers = 3
    model = Seq2SeqCLP(hidden_dim=model_hidden_dim, n_layers=model_n_layers)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Đã nạp mô hình clp_transformer.pth thành công! hidden_dim={model_hidden_dim}, n_layers={model_n_layers}")
except Exception as e:
    model_error = str(e)
    model = None
    print("❌ LỖI tải mô hình.")
    print(f"Chi tiết: {e}")

def get_ai_prediction_autoregressive(items):
    """Sử dụng Transformer dự đoán TỪNG BƯỚC MỘT (Autoregressive)"""
    assert model is not None, "Model must be loaded before prediction"
    input_seq = [[item.l, item.w, item.h] for item in items]
    src_tensor = normalize(torch.tensor([input_seq], dtype=torch.float32))
    
    tgt_seq = [[0.0, 0.0, 0.0]]
    tgt_tensor = normalize(torch.tensor([tgt_seq], dtype=torch.float32))
    
    with torch.no_grad():
        for _ in range(len(items)):
            predictions = model(src_tensor, tgt_tensor)
            next_item = predictions[:, -1:, :]
            tgt_tensor = torch.cat([tgt_tensor, next_item], dim=1)
            
    pred_seq = denormalize(tgt_tensor[0, 1:]).cpu().numpy().tolist()
    return pred_seq

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            "error": "Model chưa sẵn sàng.",
            "details": model_error
        }), 503

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
