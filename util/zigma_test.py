import torch
import torch.nn as nn

from zigMa import get_zigma_paths, ZigMaType


# 定义 PatchEmbeddingWithPosition 类
class PathEmbeddingT(nn.Module):
    def __init__(self, img_size=9, patch_size=3, stride=3, in_channels=1, embed_dim=1, num_patches=None,
                 norm_layer=None, flatten=True):
        super(PathEmbeddingT, self).__init__()

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1] if num_patches is None else num_patches
        self.flatten = flatten

        # Patch Embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, restore_original=False):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input img size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"

        # 计算 Patch Embedding
        x = self.proj(x)  # B, embed_dim, grid_h, grid_w
        print('original input')
        print(x)
        print('---------------------------------')

        zigma_order, inverse_zigma_order = get_zigma_paths(x, ZigMaType.UpToDownZig)
        if self.flatten:
            grid_h, grid_w = x.shape[2], x.shape[3]
            x = x.permute(0, 2, 3, 1)  # B, grid_h, grid_w, embed_dim
            # 拉平并按 Z 字形顺序重排
            x = x.reshape(B, grid_h * grid_w, -1)  # B, num_patches, embed_dim
            x = x[:, zigma_order, :]  # 按照 zigma 顺序重排

        # 加入位置编码
        x = x + self.pos_embed  # B, num_patches, embed_dim

        if restore_original:
            # 还原到原始顺序
            x = x[:, inverse_zigma_order, :]  # 使用逆映射还原顺序


        # 归一化
        x = self.norm(x)
        return x

# 测试 Demo
def test_patch_embedding_with_position():
    # 输入的图像
    input_tensor = torch.tensor(
        [
            [[
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]]
        ], dtype=torch.float32
    )  # Shape: (1, 1, 3, 3)

    print("输入图片:")
    print(input_tensor.squeeze(0).squeeze(0).numpy())

    # 定义模型
    model = PathEmbeddingT(img_size=3, patch_size=1, stride=1, in_channels=1, embed_dim=1, flatten=True)

    # 获取 Z 字形排列结果
    output_zigzag = model(input_tensor)
    print("\nZ 字形排列结果:")
    print(output_zigzag.squeeze(-1).squeeze(0).detach().numpy())  # 输出形状 (num_patches, embed_dim)

    # 获取恢复后的原始顺序结果
    output_restored = model(input_tensor, restore_original=True)
    print("\n恢复到原始顺序:")
    print(output_restored.squeeze(-1).squeeze(0).detach().numpy())

# 运行测试
test_patch_embedding_with_position()
