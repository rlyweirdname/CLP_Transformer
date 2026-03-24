# Đồ họa
import plotly.graph_objects as go

def draw_3d_container(container):
    """
    Sử dụng Plotly để vẽ không gian 3D của Container và các kiện hàng bên trong.
    """
    fig = go.Figure()

    def add_box(x, y, z, dx, dy, dz, color, name, opacity=0.9):
        X = [x, x+dx, x+dx, x, x, x+dx, x+dx, x]
        Y = [y, y, y+dy, y+dy, y, y, y+dy, y+dy]
        Z = [z, z, z, z, z+dz, z+dz, z+dz, z+dz]
        I = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        J = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        K = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

        fig.add_trace(go.Mesh3d(
            x=X, y=Y, z=Z, i=I, j=J, k=K,
            color=color, opacity=opacity,
            name=name, flatshading=True
        ))

    # Vẽ khung Container
    add_box(0, 0, 0, container.L, container.W, container.H, 'rgba(200, 200, 200, 0.1)', 'Container', opacity=0.1)

    # Vẽ các khối hàng
    for item in container.packed_items:
        add_box(item.x, item.y, item.z, item.l, item.w, item.h, item.color, f'Kiện {item.id}')

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Chiều dài (L)', range=[0, container.L]),
            yaxis=dict(title='Chiều rộng (W)', range=[0, container.W]),
            zaxis=dict(title='Chiều cao (H)', range=[0, container.H]),
            aspectmode='data'
        ),
        title='Mô phỏng Xếp Container 3D - Phân tích và Thiết kế thuật toán',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()