import json
import random
from environment import Item, Container
from algorithms import greedy_pack

def generate_random_problem(num_items):
    """Sinh ngẫu nhiên một container và danh sách kiện hàng."""
    # Kích thước container cố định (hoặc bạn có thể cho random)
    L, W, H = 20, 15, 10
    container = Container(L, W, H)
    
    items = []
    for i in range(num_items):
        # Sinh kích thước ngẫu nhiên cho từng kiện hàng
        l = random.randint(2, 5)
        w = random.randint(2, 4)
        h = random.randint(2, 4)
        items.append(Item(id=i, l=l, w=w, h=h))
        
    return container, items

def generate_dataset(num_samples=1000, items_per_sample=20, filename="clp_dataset.json"):
    """Chạy thuật toán Greedy nhiều lần để tạo bộ dữ liệu huấn luyện."""
    dataset = []
    
    print(f"Đang tiến hành tạo {num_samples} mẫu dữ liệu. Vui lòng đợi...")
    
    for _ in range(num_samples):
        container, items = generate_random_problem(items_per_sample)
        
        # 1. Lưu lại trạng thái ban đầu (Input) - Chỉ lấy kích thước [l, w, h]
        input_seq = [[item.l, item.w, item.h] for item in items]
        
        # 2. Giải bài toán bằng thuật toán Tham lam (Lấy đáp án)
        greedy_pack(container, items)
        
        # 3. Rút trích đáp án (Target) 
        # Là danh sách các kiện hàng MÀ THUẬT TOÁN ĐÃ XẾP THÀNH CÔNG, theo đúng thứ tự
        target_seq = [[item.l, item.w, item.h] for item in container.packed_items]
        
        # Chỉ lưu những mẫu có xếp được hàng (để tránh nhiễu dữ liệu)
        if len(target_seq) > 0:
            dataset.append({
                "input": input_seq,
                "target": target_seq
            })
            
    # Ghi toàn bộ dữ liệu ra file JSON
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"✅ Đã tạo thành công {len(dataset)} mẫu dữ liệu và lưu vào file '{filename}'")

if __name__ == "__main__":
    # Tạo thử 1000 bài toán mẫu, mỗi bài có 20 kiện hàng
    # Quá trình này chạy rất nhanh, chỉ mất vài giây
    generate_dataset(num_samples=1000, items_per_sample=20)