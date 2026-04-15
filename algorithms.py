# Chứa các thuật toán xếp hàng
def get_rotations(l, w, h):
    rotations = [
        (l, w, h), (l, h, w),
        (w, l, h), (w, h, l),
        (h, l, w), (h, w, l)
    ]
    return list(set(rotations))


def simulate_greedy_placement(container, state, dims):
    l, w, h = dims
    current_x = state["x"]
    current_y = state["y"]
    current_z = state["z"]
    row_max_w = state["row_max_w"]
    layer_max_h = state["layer_max_h"]

    if current_x + l > container.L:
        current_x = 0
        current_y += row_max_w
        row_max_w = 0

    if current_y + w > container.W:
        current_x = 0
        current_y = 0
        current_z += layer_max_h
        row_max_w = 0
        layer_max_h = 0

    if current_z + h > container.H:
        return None

    return {
        "place": {"x": current_x, "y": current_y, "z": current_z, "l": l, "w": w, "h": h},
        "next_state": {
            "x": current_x + l,
            "y": current_y,
            "z": current_z,
            "row_max_w": max(row_max_w, w),
            "layer_max_h": max(layer_max_h, h)
        }
    }


def choose_best_rotation(container, item, state):
    best = None

    for dims in get_rotations(item.l, item.w, item.h):
        sim = simulate_greedy_placement(container, state, dims)
        if sim is None:
            continue
        p = sim["place"]
        score = (
            p["z"],
            p["y"],
            p["x"],
            container.H - (p["z"] + p["h"]),
            container.W - (p["y"] + p["w"]),
            container.L - (p["x"] + p["l"])
        )

        if best is None or score < best["score"]:
            best = {"score": score, "sim": sim}

    return None if best is None else best["sim"]


def greedy_pack(container, items):
    """
    Thuật toán Tham lam (Baseline): Xếp tuần tự các kiện hàng lớn nhất trước.
    """
    state = {"x": 0, "y": 0, "z": 0, "row_max_w": 0, "layer_max_h": 0}

    # Heuristic: Ưu tiên thể tích lớn
    items.sort(key=lambda item: item.l * item.w * item.h, reverse=True)

    for item in items:
        best = choose_best_rotation(container, item, state)
        if best is None:
            print(f"Bỏ qua kiện {item.id}: Không còn tư thế xoay hợp lệ.")
            continue

        # Gán tọa độ khi thỏa mãn
        place = best["place"]
        item.x = place["x"]
        item.y = place["y"]
        item.z = place["z"]
        item.l = place["l"]
        item.w = place["w"]
        item.h = place["h"]
        item.is_packed = True

        # Thêm vào container
        container.add_item(item)

        # Cập nhật con trỏ tọa độ
        state = best["next_state"]
