import random
from environment import Item, Container
from algorithms import greedy_pack
from visualization import draw_3d_container

def main():
    # 1. Khởi tạo môi trường
    my_container = Container(length=20, width=15, height=10)
    
    # Tạo ngẫu nhiên 20 kiện hàng
    items = []
    for i in range(20):
        l = random.randint(2, 5)
        w = random.randint(2, 4)
        h = random.randint(2, 4)
        items.append(Item(id=i, l=l, w=w, h=h))
        
    print(f"Bắt đầu xếp {len(items)} kiện hàng...")
    
    # 2. Chạy thuật toán
    greedy_pack(my_container, items)
    
    print(f"Đã xếp thành công: {len(my_container.packed_items)} kiện hàng.")
    
    # 3. Trực quan hóa kết quả
    draw_3d_container(my_container)

if __name__ == "__main__":
    main()