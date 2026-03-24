import json
import random

# Tạo một lớp Box đơn giản để quản lý cả kích thước lẫn Tọa độ cắt
class Box:
    def __init__(self, x, y, z, l, w, h):
        self.x = x
        self.y = y
        self.z = z
        self.l = l
        self.w = w
        self.h = h

def generate_perfect_packing(L, W, H, num_items, min_size=2):
    """
    Kỹ thuật Guillotine Split: 
    Bắt đầu với 1 khối container đặc ruột. Cắt đệ quy thành các khối nhỏ.
    """
    # Khởi tạo container ban đầu
    boxes = [Box(0, 0, 0, L, W, H)]
    
    while len(boxes) < num_items:
        # 1. Tìm tất cả các hộp hiện tại còn đủ độ lớn để cắt tiếp
        splittable_boxes = []
        for i, b in enumerate(boxes):
            if b.l >= min_size * 2 or b.w >= min_size * 2 or b.h >= min_size * 2:
                splittable_boxes.append(i)
        
        # Nếu không còn hộp nào cắt được nữa (toàn hộp quá bé) thì dừng sớm
        if not splittable_boxes:
            break
            
        # 2. Chọn ngẫu nhiên 1 hộp để cắt
        box_idx = random.choice(splittable_boxes)
        b = boxes.pop(box_idx)
        
        # 3. Chọn ngẫu nhiên trục cắt (0: Chiều dài L, 1: Chiều rộng W, 2: Chiều cao H)
        valid_axes = []
        if b.l >= min_size * 2: valid_axes.append(0)
        if b.w >= min_size * 2: valid_axes.append(1)
        if b.h >= min_size * 2: valid_axes.append(2)
        
        axis = random.choice(valid_axes)
        
        # 4. Thực hiện nhát cắt chia khối b thành box1 và box2
        if axis == 0: # Cắt dọc theo L
            split_point = random.randint(min_size, b.l - min_size)
            box1 = Box(b.x, b.y, b.z, split_point, b.w, b.h)
            box2 = Box(b.x + split_point, b.y, b.z, b.l - split_point, b.w, b.h)
        elif axis == 1: # Cắt ngang theo W
            split_point = random.randint(min_size, b.w - min_size)
            box1 = Box(b.x, b.y, b.z, b.l, split_point, b.h)
            box2 = Box(b.x, b.y + split_point, b.z, b.l, b.w - split_point, b.h)
        else: # Cắt ngang theo H
            split_point = random.randint(min_size, b.h - min_size)
            box1 = Box(b.x, b.y, b.z, b.l, b.w, split_point)
            box2 = Box(b.x, b.y, b.z + split_point, b.l, b.w, b.h - split_point)
            
        # Thêm 2 khối mới vào danh sách
        boxes.extend([box1, box2])
        
    return boxes

def generate_optimal_dataset(num_samples=2000, items_per_sample=20, filename="clp_dataset_optimal.json"):
    L, W, H = 20, 15, 10
    dataset = []
    
    print(f"Đang dùng kỹ thuật Cắt bánh tạo {num_samples} mẫu dữ liệu TỐI ƯU...")
    
    for _ in range(num_samples):
        # Tạo ra các kiện hàng vừa khít nhau 100%
        boxes = generate_perfect_packing(L, W, H, items_per_sample)
        
        # BƯỚC ĐỘT PHÁ: Tạo đáp án (Target)
        # Để dễ xếp, kiện hàng nào ở dưới thấp (z nhỏ), góc trong cùng (y nhỏ, x nhỏ) sẽ phải được bốc ra xếp trước!
        boxes.sort(key=lambda b: (b.z, b.y, b.x))
        
        # Rút trích chỉ lấy kích thước (l, w, h) theo đúng thứ tự đã sort
        target_seq = [[b.l, b.w, b.h] for b in boxes]
        
        # BƯỚC ĐỘT PHÁ: Tạo đề bài (Input)
        # Đảo lộn xộn danh sách đi để làm đề bài đố AI
        input_seq = target_seq.copy()
        random.shuffle(input_seq)
        
        dataset.append({
            "input": input_seq,
            "target": target_seq
        })
        
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)
    print(f"✅ Xong! Đã lưu dữ liệu vào '{filename}'. Tỷ lệ lấp đầy mỗi mẫu đều là 100%!")

if __name__ == "__main__":
    generate_optimal_dataset(num_samples=2000, items_per_sample=50)