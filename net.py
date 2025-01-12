# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:35:21 2020

@author: ZJU
"""

import torch
import torch.nn as nn


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std

# 输入为[batch_size, 512, 32, 32], 输出为[batch_size, 3, 256, 256]
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),  # 通道数从 512 -> 256
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),  # 空间尺寸 32x32 -> 64x64

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),  # 通道数从 256 -> 128
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),  # 空间尺寸 64x64 -> 128x128

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),  # 通道数从 128 -> 64
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),  # 空间尺寸 128x128 -> 256x256

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 32, (3, 3)),  # 通道数从 64 -> 32
    nn.ReLU(),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(32, 16, (3, 3)),  # 通道数从 32 -> 16
    nn.ReLU(),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(16, 3, (3, 3)),  # 通道数从 16 -> 3（RGB 图像）
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

projection_style = nn.Sequential(
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=128)
)

projection_content = nn.Sequential(
    nn.Linear(in_features=512, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=128)
)


class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs

class Net(nn.Module):
    def __init__(self, vgg, mamba, decoder, feature_is_mixed, patch_embedding_1, patch_embedding_2,
                 patch_decoder_1, patch_decoder_2, weight_1, weight_2, start_iter):
        super(Net, self).__init__()
        enc_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        #projection
        self.proj_style = projection_style
        self.proj_content = projection_content
        self.feature_is_mixed = feature_is_mixed

        #patch embedding and path_embedding_reverse
        self.patch_embedding_1 = patch_embedding_1
        self.patch_embedding_reverse_1 = patch_decoder_1

        self.patch_embedding_2 = patch_embedding_2
        self.patch_embedding_reverse_2 = patch_decoder_2

        self.weight_1 = weight_1
        self.weight_2 = weight_2

        # mamba
        self.mamba = mamba
        self.decoder = decoder
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if(start_iter > 0):
            self.decoder.load_state_dict(torch.load('decoder_iter_' + str(start_iter) + '.pth'))
            self.mamba.load_state_dict(torch.load('mamba_iter_' + str(start_iter) + '.pth'))
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target, norm = False):
        if(norm == False):
          return self.mse_loss(input, target)
        else:
          return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def compute_contrastive_loss(self, feat_q, feat_k, tau, index):
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / tau
        #loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        loss = self.cross_entropy_loss(out, torch.tensor([index], dtype=torch.long, device=feat_q.device))
        return loss

    def style_feature_contrastive(self, input):
        # out = self.enc_style(input)
        out = torch.sum(input, dim=[2, 3])
        out = self.proj_style(out)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out

    def content_feature_contrastive(self, input):
        #out = self.enc_content(input)
        out = torch.sum(input, dim=[2, 3])
        out = self.proj_content(out)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out

    def forward(self, content):
        # step 1. 使用mamba + decoder，对content图片进行编解码重建
        # branch1
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        patch_embed_content_1 = self.patch_embedding_1(content)
        # [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        encoded_content_1 = self.mamba(patch_embed_content_1)
        # [B, num_patches, embed_dim] -> [B, embed_dim, num_patches] -> [B, embed_dim, grid_h, grid_w]
        decoded_content_1 = self.patch_embedding_reverse_1(
            encoded_content_1, self.patch_embedding_1.pos_embed, self.patch_embedding_1.inverse_zigma_order)
        # [B, embed_dim, grid_h, grid_w] -> [B, C, H, W]
        g_content_1 = self.decoder(decoded_content_1)

        if self.feature_is_mixed:
            # branch2
            # [B, C, H, W] -> [B, num_patches, embed_dim]
            patch_embed_content_2 = self.patch_embedding_2(content)
            # [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
            encoded_content_2 = self.mamba(patch_embed_content_2)
            # [B, num_patches, embed_dim] -> [B, embed_dim, num_patches] -> [B, embed_dim, grid_h, grid_w]
            decoded_content_2 = self.patch_embedding_reverse_2(
                encoded_content_2, self.patch_embedding_2.pos_embed, self.patch_embedding_2.inverse_zigma_order)
            # [B, embed_dim, grid_h, grid_w] -> [B, C, H, W]
            g_content_2 = self.decoder(decoded_content_2)

            g_content = self.weight_1 * g_content_1 + self.weight_2 * g_content_2
        else:
            g_content = g_content_1

        content_feats = self.encode_with_intermediate(content)
        g_c_feats = self.encode_with_intermediate(g_content)
        loss_c = self.calc_content_loss(g_c_feats[0], content_feats[0])
        for i in range(1, 5):
            loss_c += self.calc_content_loss(content_feats[i], g_c_feats[i])
        return g_content, loss_c