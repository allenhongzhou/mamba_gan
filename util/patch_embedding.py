import torch
import torch.nn as nn
import numpy as np

from util.zigMa import ZigMaType, get_zigma_paths


class PatchEmbeddingWithPosition(nn.Module):
    def __init__(self, img_size=256, patch_size=8, stride=8, in_channels=3, embed_dim=512, zig_type=ZigMaType.ZigMaNone,
                 num_patches=None, norm_layer=None, flatten=True):
        super(PatchEmbeddingWithPosition, self).__init__()

        # 原有初始化
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1] if num_patches is None else num_patches
        self.flatten = flatten
        self.zigType = zig_type
        self.zigma_order = []
        self.inverse_zigma_order =[]

        # Patch Embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride)

        # 初始化位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))  # 可学习的

        # 可选的归一化层
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input img size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"

        # 计算 Patch Embedding
        x = self.proj(x)  # B, embed_dim, grid_h, grid_w
        self.zigma_order, self.inverse_zigma_order = get_zigma_paths(x, self.zigType)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B, embed_dim, grid_h*grid_w -> B, num_patches, embed_dim
            if self.zigType != ZigMaType.ZigMaNone:
                x = x = x[:, self.zigma_order, :]  # 按照 zigma 顺序重排

        # 将位置编码添加到 patch embedding 上
        x = x + self.pos_embed  # B, num_patches, embed_dim

        # 归一化
        x = self.norm(x)
        return x
