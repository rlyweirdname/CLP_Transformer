# Chứa các thuật toán xếp hàng
def greedy_pack(container, items):
    """
    Thuật toán Tham lam (Baseline): Xếp tuần tự các kiện hàng lớn nhất trước.
    """
    current_x, current_y, current_z = 0, 0, 0
    max_h_in_row = 0
    
    # Heuristic: Ưu tiên thể tích lớn
    items.sort(key=lambda item: item.l * item.w * item.h, reverse=True)
    
    for item in items:
        # Check tràn chiều dài L
        if current_x + item.l > container.L:
            current_x = 0
            current_y += item.w
        
        # Check tràn chiều rộng W
        if current_y + item.w > container.W:
            current_x = 0
            current_y = 0
            current_z += max_h_in_row
            max_h_in_row = 0
            
        # Check tràn chiều cao H
        if current_z + item.h > container.H:
            print(f"Bỏ qua kiện {item.id}: Đã đầy chiều cao.")
            continue
            
        # Gán tọa độ khi thỏa mãn
        item.x = current_x
        item.y = current_y
        item.z = current_z
        item.is_packed = True
        
        # Thêm vào container
        container.add_item(item)
        
        # Cập nhật con trỏ tọa độ
        current_x += item.l
        max_h_in_row = max(max_h_in_row, item.h)