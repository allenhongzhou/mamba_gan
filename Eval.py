import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import net
from mamba_module import SingleMambaBlock
from util.patch_embedding import PatchEmbeddingWithPosition
from util.patch_embedding_reverse import PatchDecoder
from util.zigMa import ZigMaType


def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def test_transform1():
    transform_list = [
        transforms.Resize((256, 256)),
        #transforms.CenterCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str, default='input/content/12.jpg',
                    help='File path to the content image')
parser.add_argument('--steps', type=str, default=1)
parser.add_argument('--decoder', type=str, default='./experiments/decoder_iter_160000.pth')
parser.add_argument('--mamba', type=str, default='./experiments/mamba_iter_160000.pth')
parser.add_argument('--patch_embedding1', type=str, default='./experiments/patch_embedding_1_iter_160000.pth')
parser.add_argument('--patch_decoder1', type=str, default='./experiments/patch_embedding_reverse_1_iter_160000.pth')
parser.add_argument('--patch_embedding2', type=str, default='./experiments/patch_embedding_2_iter_160000.pth')
parser.add_argument('--patch_decoder2', type=str, default='./experiments/patch_embedding_reverse_2_iter_160000.pth')

# Additional options
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = net.decoder
mamba = SingleMambaBlock(dim=512)
decoder.eval()
mamba.eval()
decoder.load_state_dict(torch.load(args.decoder))
mamba.load_state_dict(torch.load(args.mamba))
decoder.to(device)
mamba.to(device)

patch_embedding_1 = PatchEmbeddingWithPosition(zig_type=ZigMaType.UpToDownZig)
patch_decoder_1 = PatchDecoder()
patch_embedding_1.eval()
patch_decoder_1.eval()
patch_embedding_1.load_state_dict((torch.load(args.patch_embedding1)))
patch_decoder_1.load_state_dict((torch.load(args.patch_decoder1)))
patch_embedding_1.to(device)
patch_decoder_1.to(device)

weight_1, weight_2 = 0.5, 0.5
patch_embedding_2, patch_decoder_2 = None, None

feature_is_mixed = True
if feature_is_mixed:
    weight_1 = torch.tensor(weight_1, device=device)
    weight_2 = torch.tensor(weight_2, device=device)
    patch_embedding_2 = PatchEmbeddingWithPosition(zig_type=ZigMaType.LeftToRightZig)
    patch_decoder_2 = PatchDecoder()
    patch_embedding_2.eval()
    patch_decoder_2.eval()
    patch_embedding_2.load_state_dict((torch.load(args.patch_embedding2)))
    patch_decoder_2.load_state_dict((torch.load(args.patch_decoder2)))
    patch_embedding_2.to(device)
    patch_decoder_2.to(device)

# TODO 理解cycle-gan的测试的流程
content_tf = test_transform1()
content = content_tf(Image.open(args.content))
content = content.to(device).unsqueeze(0)

with torch.no_grad():
    for x in range(args.steps):
        print('iteration ' + str(x))
        patch_embed_content_1 = patch_embedding_1(content)
        # [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        encoded_content_1 = mamba(patch_embed_content_1)
        # [B, num_patches, embed_dim] -> [B, embed_dim, num_patches] -> [B, embed_dim, grid_h, grid_w]
        decoded_content_1 = patch_decoder_1(
            encoded_content_1, patch_embedding_1.pos_embed, patch_embedding_1.inverse_zigma_order)
        # [B, embed_dim, grid_h, grid_w] -> [B, C, H, W]
        g_content_1 = decoder(decoded_content_1)
        g_content = g_content_1
        if feature_is_mixed:
            patch_embed_content_2 = patch_embedding_2(content)
            # [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
            encoded_content_2 = mamba(patch_embed_content_2)
            # [B, num_patches, embed_dim] -> [B, embed_dim, num_patches] -> [B, embed_dim, grid_h, grid_w]
            decoded_content_2 = patch_decoder_2(
                encoded_content_2, patch_embedding_2.pos_embed, patch_embedding_2.inverse_zigma_order)
            # [B, embed_dim, grid_h, grid_w] -> [B, C, H, W]
            g_content_2 = decoder(decoded_content_1)
            g_content = weight_1 * g_content + weight_2 * g_content_2

    g_content.clamp(0, 255)
    g_content = g_content.cpu()
    output_name = '{:s}/{:s}_test1_{:s}{:s}'.format(
        args.output, splitext(basename(args.content))[0],
        splitext(basename(args.content))[0], args.save_ext
    )
    save_image(g_content, output_name)
