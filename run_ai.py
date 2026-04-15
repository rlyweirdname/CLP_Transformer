import torch
import random
from environment import Item, Container
from visualization import draw_3d_container
from model import Seq2SeqCLP

def extract_state_dict(raw_checkpoint):
    if isinstance(raw_checkpoint, dict) and "state_dict" in raw_checkpoint:
        return raw_checkpoint["state_dict"]
    return raw_checkpoint

def infer_hidden_dim(state_dict, default_dim=128):
    emb_weight = state_dict.get("embedding.weight")
    if emb_weight is not None and hasattr(emb_weight, "shape") and len(emb_weight.shape) == 2:
        return int(emb_weight.shape[0])
    return default_dim

def infer_n_layers(state_dict, default_n_layers=3):
    layer_indices = set()
    for key in state_dict.keys():
        parts = key.split(".")
        if "layers" in parts:
            idx = parts.index("layers")
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                layer_indices.add(int(parts[idx + 1]))
    return max(layer_indices) + 1 if layer_indices else default_n_layers


def beam_match_items_to_predictions(items, predicted_seq, beam_width=4):
    """Ghép item theo chuỗi dự đoán bằng Beam Search để giảm sai số tích lũy."""
    if not items or len(predicted_seq) == 0:
        return items.copy()

    beams = [{"score": 0.0, "order": [], "used": frozenset()}]

    for pred_dims in predicted_seq:
        next_beams = []
        for beam in beams:
            for idx, it in enumerate(items):
                if idx in beam["used"]:
                    continue
                diff = abs(it.l - pred_dims[0]) + abs(it.w - pred_dims[1]) + abs(it.h - pred_dims[2])
                next_beams.append({
                    "score": beam["score"] + diff,
                    "order": beam["order"] + [idx],
                    "used": beam["used"] | {idx}
                })

        if len(next_beams) == 0:
            break
        next_beams.sort(key=lambda x: x["score"])
        beams = next_beams[:beam_width]

    if len(beams) == 0:
        return items.copy()

    best = beams[0]
    ordered = [items[idx] for idx in best["order"]]
    for idx, it in enumerate(items):
        if idx not in best["used"]:
            ordered.append(it)
    return ordered


def get_ai_prediction(model, items):
    """Sử dụng Transformer để dự đoán TỪNG BƯỚC MỘT (Autoregressive)."""
    model.eval() 
    
    # 1. Chuyển danh sách kiện hàng thành Tensor (Input)
    input_seq = [[item.l, item.w, item.h] for item in items]
    src_tensor = torch.tensor([input_seq], dtype=torch.float32)
    
    # 2. Khởi tạo mảng Target với 1 kiện hàng "giả" (Tọa độ 0) để làm mồi nhử (Start Token)
    tgt_seq = [[0.0, 0.0, 0.0]]
    tgt_tensor = torch.tensor([tgt_seq], dtype=torch.float32)
    
    with torch.no_grad():
        # Vòng lặp Tự hồi quy: Dự đoán từng kiện hàng một
        for _ in range(len(items)):
            predictions = model(src_tensor, tgt_tensor)
            
            # Lấy dự đoán của bước cuối cùng vừa sinh ra
            next_item = predictions[:, -1:, :]
            
            # Nối (Concat) dự đoán mới vào mảng Target để làm manh mối cho bước tiếp theo
            tgt_tensor = torch.cat([tgt_tensor, next_item], dim=1)
    
    # Loại bỏ token mồi nhử [0,0,0] ban đầu, chỉ lấy các dự đoán thực sự
    pred_seq = tgt_tensor[0, 1:].detach().cpu().numpy()
    return pred_seq

def check_overlap(item1, item2):
    """Kiểm tra đụng độ 3D giữa 2 kiện hàng."""
    return not (item1.x + item1.l <= item2.x or
                item1.x >= item2.x + item2.l or
                item1.y + item1.w <= item2.y or
                item1.y >= item2.y + item2.w or
                item1.z + item1.h <= item2.z or
                item1.z >= item2.z + item2.h)

def get_rotations(l, w, h):
    """Trả về danh sách 6 hướng xoay có thể có của một kiện hàng (loại bỏ các hướng trùng lặp)."""
    rotations = [
        (l, w, h), (l, h, w), 
        (w, l, h), (w, h, l), 
        (h, l, w), (h, w, l)
    ]
    # Dùng set để loại bỏ các cấu hình trùng nhau (nếu kiện hàng có hình vuông)
    return list(set(rotations))

def ai_pack(container, items, predicted_seq):
    """Xếp hàng thông minh: Kết hợp Extreme Points + Xoay 6 hướng."""
    items_to_pack = beam_match_items_to_predictions(items, predicted_seq, beam_width=4)
        
    # 2. Danh sách các tọa độ điểm cực trị (Bắt đầu từ gốc 0,0,0)
    valid_points = [(0, 0, 0)]
    
    for item in items_to_pack:
        best_point = None
        best_rotation = None
        
        # Luôn ưu tiên quét các góc thấp nhất (Z), sâu nhất (Y), trái nhất (X)
        valid_points.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for pt in valid_points:
            # THỬ CẢ 6 HƯỚNG XOAY TẠI ĐIỂM NÀY
            for rot_l, rot_w, rot_h in get_rotations(item.l, item.w, item.h):
                item.x, item.y, item.z = pt
                
                # Check 1: Tràn container?
                if item.x + rot_l > container.L or item.y + rot_w > container.W or item.z + rot_h > container.H:
                    continue
                    
                # Check 2: Chồng lấn với kiện đã xếp?
                # Tạm thời gán kích thước xoay để test overlap
                temp_l, temp_w, temp_h = item.l, item.w, item.h
                item.l, item.w, item.h = rot_l, rot_w, rot_h
                
                overlap = False
                for packed in container.packed_items:
                    if check_overlap(item, packed):
                        overlap = True
                        break
                
                if not overlap:
                    best_point = pt
                    best_rotation = (rot_l, rot_w, rot_h)
                    break # Tìm được góc và hướng xoay hợp lý -> Dừng thử xoay
                else:
                    # Hoàn trả kích thước cũ nếu xoay bị lỗi
                    item.l, item.w, item.h = temp_l, temp_w, temp_h
            
            if best_point:
                break # Đã chốt được điểm -> Chuyển sang kiện hàng tiếp theo
                
        # 3. Chốt hạ việc đặt kiện hàng
        if best_point:
            assert best_rotation is not None
            item.l, item.w, item.h = best_rotation
            item.is_packed = True
            container.add_item(item)
            
            # 4. Sinh ra 3 điểm cực trị mới (Extreme Points) từ các nóc/cạnh của kiện vừa đặt
            valid_points.append((item.x + item.l, item.y, item.z))
            valid_points.append((item.x, item.y + item.w, item.z))
            valid_points.append((item.x, item.y, item.z + item.h))

def main():
    print("Khởi tạo môi trường...")
    container = Container(length=20, width=15, height=10)
    
    items = []
    for i in range(80):
        items.append(Item(id=i, l=random.randint(2, 5), w=random.randint(2, 4), h=random.randint(2, 4)))
        
    print("Đang tải mô hình Transformer đã huấn luyện...")
    try:
        raw_checkpoint = torch.load("clp_transformer.pth", map_location="cpu")
        state_dict = extract_state_dict(raw_checkpoint)
        hidden_dim = infer_hidden_dim(state_dict, default_dim=128)
        n_layers = 3
        model = Seq2SeqCLP(hidden_dim=hidden_dim, n_layers=n_layers)
        model.load_state_dict(state_dict)
        print(f"Tải mô hình thành công! hidden_dim={hidden_dim}, n_layers={n_layers}")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file clp_transformer.pth. Hãy chạy train.py trước.")
        return
    except Exception as e:
        print("Lỗi: Không thể tải checkpoint.")
        print(f"Chi tiết: {e}")
        return

    print("AI đang tính toán chuỗi sắp xếp tối ưu...")
    predicted_seq = get_ai_prediction(model, items)
    
    print("Đang gán tọa độ và đưa vào Container...")
    ai_pack(container, items, predicted_seq)
    
    print(f"Hoàn tất! AI đã xếp được {len(container.packed_items)}/{len(items)} kiện hàng.")
    
    draw_3d_container(container)

if __name__ == "__main__":
    main()
