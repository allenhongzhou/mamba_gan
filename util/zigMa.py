from enum import Enum

class ZigMaType(Enum):
    LeftToRightZig = 1 # 从上到下，从下到上，zig扫描
    UpToDownZig = 2 # 从左到右，从右到左，zig扫描
    ZigMaNone = 3

def get_zigma_paths(x, zigMaType):
    # 输入的x维度是B, embed_dim, grid_h, grid_w
    B, C, H, W = x.shape
    grid_h, grid_w = x.shape[2], x.shape[3]
    # 调整维度顺序，方便操作
    x = x.permute(0, 2, 3, 1)  # B, grid_h, grid_w, embed_dim
    zigma_order = []
    if zigMaType == ZigMaType.UpToDownZig:
        # Z 字形扫描调整（列优先）
        for j in range(grid_w):
            if j % 2 == 0:
                zigma_order.extend([i * grid_w + j for i in range(grid_h)])  # 从上到下
            else:
                zigma_order.extend([i * grid_w + j for i in range(grid_h - 1, -1, -1)])  # 从下到上

        # 创建逆映射
        inverse_zigma_order = [0] * len(zigma_order)
        for idx, pos in enumerate(zigma_order):
            inverse_zigma_order[pos] = idx
        # 拉平并按 Z 字形顺序重排
        # x = x.reshape(B, grid_h * grid_w, -1)  # B, num_patches, embed_dim
        # x = x[:, zigma_order, :]  # 按照 zigma 顺序重排
        return zigma_order, inverse_zigma_order

    if zigMaType == ZigMaType.LeftToRightZig:
        # Z 字形扫描调整
        for i in range(grid_h):
            if i % 2 == 0:
                zigma_order.extend(range(i * grid_w, (i + 1) * grid_w))  # 从左到右
            else:
                zigma_order.extend(range((i + 1) * grid_w - 1, i * grid_w - 1, -1))  # 从右到左
        # 创建逆映射
        inverse_zigma_order = [0] * len(zigma_order)
        for i, idx in enumerate(zigma_order):
            inverse_zigma_order[idx] = i
        # 拉平并按 Z 字形顺序重排
        # x = x.reshape(B, grid_h * grid_w, -1)  # B, num_patches, embed_dim
        # x = x[:, zigma_order, :]  # 按照 zigma 顺序重排
        return zigma_order, inverse_zigma_order


