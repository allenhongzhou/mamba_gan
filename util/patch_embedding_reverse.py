import torch
import torch.nn as nn


class PatchDecoder(nn.Module):
    def __init__(self, img_size=256, patch_size=8, stride=8, in_channels=3, embed_dim=512, norm_layer=None):
        super(PatchDecoder, self).__init__()

        # 保存相关参数
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        # 可选的归一化层（如果编码器中使用了，解码器中也需要对应）
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # # 逆卷积（反向 Patch Embedding）
        # self.proj = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=patch_size, stride=stride)

    def forward(self, x, pos_embed=None, inverse_zigma_order=[]):
        """
        Args:
            x: 编码后的输入，形状为 [B, num_patches, embed_dim]
            pos_embed: 位置编码，形状为 [1, num_patches, embed_dim]

        Returns:
            还原后的图像，形状为 [B, C, H, W]
        """
        B, num_patches, embed_dim = x.shape
        grid_h = (self.img_size - self.patch_size) // self.stride + 1
        grid_w = (self.img_size - self.patch_size) // self.stride + 1
        assert num_patches == grid_h * grid_w, "编码器的 num_patches 与解码器不匹配"


        # 1. 如果有位置编码，移除它
        if pos_embed is not None:
            x = x - pos_embed  # 广播操作移除位置编码

        # 2. 逆归一化
        x = self.norm(x)

        # 3. 使用逆映射还原顺序
        if len(inverse_zigma_order) != 0:
            x = x[:, inverse_zigma_order, :]

        # 4. 将嵌入向量还原为图像块
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim] -> [B, embed_dim, num_patches]
        x = x.view(B, embed_dim, grid_h, grid_w)  # [B, embed_dim, num_patches] -> [B, embed_dim, grid_h, grid_w]

        ## 5. 使用转置卷积还原为原始图像
        # x = self.proj(x)  # [B, C, H, W]
        return x
