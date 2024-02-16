import sys

sys.path.append('.')
import os
import cv2
import argparse
import torch

from utils import show_image, normalize_image
from data_readers.kitti import KITTIFlow
import torch.nn.functional as F
from thop import profile, clever_format

import matplotlib.pyplot as plt
from data_readers.frame_utils import *


def display(img, tau, phi):
    """ display se3 fields """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(img[:, :, ::-1] / 255.0)

    tau_img = np.clip(tau, -0.1, 0.1)
    tau_img = (tau_img + 0.1) / 0.2

    phi_img = np.clip(phi, -0.1, 0.1)
    phi_img = (phi_img + 0.1) / 0.2

    ax2.imshow(tau_img)
    ax3.imshow(phi_img)
    plt.show()


def prepare_images_and_depths(image1, image2, depth1, depth2, depth_scale=1.0):
    """ padding, normalization, and scaling """

    ht, wd = image1.shape[-2:]
    pad_h = (-ht) % 8
    pad_w = (-wd) % 8

    image1 = F.pad(image1, [0, pad_w, 0, pad_h], mode='replicate')
    image2 = F.pad(image2, [0, pad_w, 0, pad_h], mode='replicate')
    depth1 = F.pad(depth1[:, None], [0, pad_w, 0, pad_h], mode='replicate')[:, 0]
    depth2 = F.pad(depth2[:, None], [0, pad_w, 0, pad_h], mode='replicate')[:, 0]

    depth1 = (depth_scale * depth1).float()
    depth2 = (depth_scale * depth2).float()
    image1 = normalize_image(image1.float())
    image2 = normalize_image(image2.float())

    depth1 = depth1.float()
    depth2 = depth2.float()

    return image1, image2, depth1, depth2, (pad_w, pad_h)


@torch.no_grad()
def eval_ops(model, args):
    test_dataset = KITTIFlow(args.root, args.list_filenames, msnet_mode=args.msnet_mode)
    DEPTH_SCALE = .1

    idx, image1, image2, disp1, disp2, intrinsics = [item.cuda() for item in test_dataset[0]]
    image1 = image1[None, ...]
    image2 = image2[None, ...]
    disp1 = disp1[None, ...]
    disp2 = disp2[None, ...]
    intrinsics = intrinsics[None, ...]

    depth1 = DEPTH_SCALE * (intrinsics[0, 0] / disp1)
    depth2 = DEPTH_SCALE * (intrinsics[0, 0] / disp2)

    ht, wd = image1.shape[2:]
    image1, image2, depth1, depth2, _ = \
        prepare_images_and_depths(image1, image2, depth1, depth2)

    macs, params = clever_format(profile(model, inputs=(image1, image2, depth1, depth2, intrinsics, 16)), '%.3f')
    print(f'MACS: {macs}, PARAMS: {params}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path the model weights')
    parser.add_argument('--network', default='raft3d.raft3d', help='network architecture')
    parser.add_argument('--radius', type=int, default=32)
    parser.add_argument('--root', type=str, default='/work/vig/Datasets/KITTI_VoxelFlow')
    parser.add_argument('--list_filenames', type=str, default='./filenames/KITTI_flow_valid.txt')
    parser.add_argument('--msnet_mode', type=str)
    args = parser.parse_args()

    import importlib

    RAFT3D = importlib.import_module(args.network).RAFT3D

    model = torch.nn.DataParallel(RAFT3D(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    eval_ops(model, args)
