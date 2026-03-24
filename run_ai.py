import torch
import random
from environment import Item, Container
from visualization import draw_3d_container
from model import Seq2SeqCLP

def get_ai_prediction(model, items):
    """Sử dụng Transformer để dự đoán thứ tự kiện hàng nên xếp."""
    model.eval() # Chuyển mô hình sang chế độ suy luận (Inference)
    
    # 1. Chuyển danh sách kiện hàng thành Tensor (Input)
    input_seq = [[item.l, item.w, item.h] for item in items]
    src_tensor = torch.tensor([input_seq], dtype=torch.float32)
    
    # 2. Tạo một Tensor rỗng làm điểm xuất phát cho Decoder (Target giả)
    tgt_tensor = torch.zeros_like(src_tensor)
    
    with torch.no_grad(): # Không tính đạo hàm để chạy nhanh hơn
        # Mớm cho AI để nó dự đoán chuỗi đầu ra
        predictions = model(src_tensor, tgt_tensor)
    
    # Lấy kết quả (batch 0)
    pred_seq = predictions[0].numpy()
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
    items_to_pack = []
    unpacked_items = items.copy()
    
    # 1. Bắt cặp kiện hàng thực tế với dự đoán của Transformer
    for pred_dims in predicted_seq:
        if len(unpacked_items) == 0:
            break
        best_match = min(unpacked_items, key=lambda i: abs(i.l - pred_dims[0]) + abs(i.w - pred_dims[1]) + abs(i.h - pred_dims[2]))
        items_to_pack.append(best_match)
        unpacked_items.remove(best_match)
        
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
    
    # Tạo ngẫu nhiên 20 kiện hàng mới tinh (AI chưa từng thấy)
    items = []
    for i in range(50):
        items.append(Item(id=i, l=random.randint(2, 5), w=random.randint(2, 4), h=random.randint(2, 4)))
        
    print("Đang tải mô hình Transformer đã huấn luyện...")
    model = Seq2SeqCLP()
    try:
        model.load_state_dict(torch.load("clp_transformer.pth"))
        print("Tải mô hình thành công!")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file clp_transformer.pth. Hãy chạy train.py trước.")
        return

    print("AI đang tính toán chuỗi sắp xếp tối ưu...")
    predicted_seq = get_ai_prediction(model, items)
    
    print("Đang gán tọa độ và đưa vào Container...")
    ai_pack(container, items, predicted_seq)
    
    print(f"Hoàn tất! AI đã xếp được {len(container.packed_items)}/{len(items)} kiện hàng.")
    
    # Vẽ mô phỏng 3D
    draw_3d_container(container)

if __name__ == "__main__":
    main()