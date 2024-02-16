
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import cv2
import math
import random
import json
import csv
import pickle
import os.path as osp

from glob import glob

from . import frame_utils
from .augmentation import RGBDAugmentor, SparseAugmentor
from .data_io import read_all_lines


class KITTIEval(data.Dataset):

    crop = 80

    def __init__(self, image_size=None, root='datasets/KITTI', do_augment=True):
        self.init_seed = None
        mode = "testing"
        self.image1_list = sorted(glob(osp.join(root, mode, "image_2/*10.png")))
        self.image2_list = sorted(glob(osp.join(root, mode, "image_2/*11.png")))
        self.disp1_ga_list = sorted(glob(osp.join(root, mode, "disp_ganet_{}/*10.png".format(mode))))
        self.disp2_ga_list = sorted(glob(osp.join(root, mode, "disp_ganet_{}/*11.png".format(mode))))
        self.calib_list = sorted(glob(osp.join(root, mode, "calib_cam_to_cam/*.txt")))

        self.intrinsics_list = []
        for calib_file in self.calib_list:
            with open(calib_file) as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if row[0] == 'K_02:':
                        K = np.array(row[1:], dtype=np.float32).reshape(3,3)
                        kvec = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
                        self.intrinsics_list.append(kvec)

    @staticmethod
    def write_prediction(index, disp1, disp2, flow):

        def writeFlowKITTI(filename, uv):
            uv = 64.0 * uv + 2**15
            valid = np.ones([uv.shape[0], uv.shape[1], 1])
            uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
            cv2.imwrite(filename, uv[..., ::-1])

        def writeDispKITTI(filename, disp):
            disp = (256 * disp).astype(np.uint16)
            cv2.imwrite(filename, disp)

        disp1 = np.pad(disp1, ((KITTIEval.crop,0),(0,0)), mode='edge')
        disp2 = np.pad(disp2, ((KITTIEval.crop, 0), (0,0)), mode='edge')
        flow = np.pad(flow, ((KITTIEval.crop, 0), (0,0),(0,0)), mode='edge')

        disp1_path = 'kitti_submission/disp_0/%06d_10.png' % index
        disp2_path = 'kitti_submission/disp_1/%06d_10.png' % index
        flow_path = 'kitti_submission/flow/%06d_10.png' % index

        writeDispKITTI(disp1_path, disp1)
        writeDispKITTI(disp2_path, disp2)
        writeFlowKITTI(flow_path, flow)
                        
    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, index):

        intrinsics = self.intrinsics_list[index]
        image1 = cv2.imread(self.image1_list[index])
        image2 = cv2.imread(self.image2_list[index])

        disp1 = cv2.imread(self.disp1_ga_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp2 = cv2.imread(self.disp2_ga_list[index], cv2.IMREAD_ANYDEPTH) / 256.0

        image1 = image1[self.crop:]
        image2 = image2[self.crop:]
        disp1 = disp1[self.crop:]
        disp2 = disp2[self.crop:]
        intrinsics[3] -= self.crop

        image1 = torch.from_numpy(image1).float().permute(2,0,1)
        image2 = torch.from_numpy(image2).float().permute(2,0,1)
        disp1 = torch.from_numpy(disp1).float()
        disp2 = torch.from_numpy(disp2).float()
        intrinsics = torch.from_numpy(intrinsics).float()

        return image1, image2, disp1, disp2, intrinsics


class KITTI(data.Dataset):
    def __init__(self, image_size=None, root='datasets/KITTI', do_augment=True):
        import csv

        self.init_seed = None
        self.crop = 80

        if do_augment:
            self.augmentor = SparseAugmentor(image_size)
        else:
            self.augmentor = None
        
        self.image1_list = sorted(glob(osp.join(root, "training", "image_2/*10.png")))
        self.image2_list = sorted(glob(osp.join(root, "training", "image_2/*11.png")))

        self.disp1_list = sorted(glob(osp.join(root, "training", "disp_occ_0/*10.png")))
        self.disp2_list = sorted(glob(osp.join(root, "training", "disp_occ_1/*10.png")))

        self.disp1_ga_list = sorted(glob(osp.join(root, "training", "disp_ganet/*10.png")))
        self.disp2_ga_list = sorted(glob(osp.join(root, "training", "disp_ganet/*11.png")))

        self.flow_list = sorted(glob(osp.join(root, "training", "flow_occ/*10.png")))
        self.calib_list = sorted(glob(osp.join(root, "training", "calib_cam_to_cam/*.txt")))

        self.intrinsics_list = []
        for calib_file in self.calib_list:
            with open(calib_file) as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if row[0] == 'K_02:':
                        K = np.array(row[1:], dtype=np.float32).reshape(3,3)
                        kvec = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
                        self.intrinsics_list.append(kvec)
                        
    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        image1 = cv2.imread(self.image1_list[index])
        image2 = cv2.imread(self.image2_list[index])

        disp1 = cv2.imread(self.disp1_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp2 = cv2.imread(self.disp2_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp1_dense = cv2.imread(self.disp1_ga_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp2_dense = cv2.imread(self.disp2_ga_list[index], cv2.IMREAD_ANYDEPTH) / 256.0

        flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        intrinsics = self.intrinsics_list[index]

        SCALE = np.random.uniform(0.08, 0.15)

        # crop top 80 pixels, no ground truth information
        image1 = image1[self.crop:]
        image2 = image2[self.crop:]
        disp1 = disp1[self.crop:]
        disp2 = disp2[self.crop:]
        flow = flow[self.crop:]
        valid = valid[self.crop:]
        disp1_dense = disp1_dense[self.crop:]
        disp2_dense = disp2_dense[self.crop:]
        intrinsics[3] -= self.crop

        image1 = torch.from_numpy(image1).float().permute(2,0,1)
        image2 = torch.from_numpy(image2).float().permute(2,0,1)

        disp1 = torch.from_numpy(disp1 / intrinsics[0]) / SCALE
        disp2 = torch.from_numpy(disp2 / intrinsics[0]) / SCALE
        disp1_dense = torch.from_numpy(disp1_dense / intrinsics[0]) / SCALE
        disp2_dense = torch.from_numpy(disp2_dense / intrinsics[0]) / SCALE

        dz = (disp2 - disp1_dense).unsqueeze(dim=-1)
        depth1 = 1.0 / disp1_dense.clamp(min=0.01).float()
        depth2 = 1.0 / disp2_dense.clamp(min=0.01).float()

        intrinsics = torch.from_numpy(intrinsics)
        valid = torch.from_numpy(valid)
        flow = torch.from_numpy(flow)

        valid = valid * (disp2 > 0).float()
        flow = torch.cat([flow, dz], -1)

        if self.augmentor is not None:
            image1, image2, depth1, depth2, flow, valid, intrinsics = \
                self.augmentor(image1, image2, depth1, depth2, flow, valid, intrinsics)

        return image1, image2, depth1, depth2, flow, valid, intrinsics


class KITTIFlow(data.Dataset):
    crop = 80

    def __init__(self, root, list_filenames, *, msnet_mode='2D'):
        self.root = root
        self.init_seed = None
        self.file_idx = None
        self.image1_list = None
        self.image2_list = None
        self.disp1_ms_list = None
        self.disp2_ms_list = None
        self.calib_list = None
        self.load_path(list_filenames, msnet_mode)

        self.intrinsics_list = []
        for calib_file in self.calib_list:
            with open(osp.join(self.root, calib_file)) as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if row[0] == 'K_02:':
                        K = np.array(row[1:], dtype=np.float32).reshape(3, 3)
                        kvec = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
                        self.intrinsics_list.append(kvec)

    def load_path(self, list_filename, msnet_mode):
        # Format - left_0, right_0, left_1, right_1, disp_0, disp_1, flow, gt_voxel_0, gt_voxel_1, gt_flow, calib
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        self.file_idx = []
        self.image1_list = []
        self.image2_list = []
        self.disp1_ms_list = []
        self.disp2_ms_list = []
        self.calib_list = []
        for x in splits:
            file_id = x[0].split('/')[-1].split('_')[0]
            self.file_idx.append(int(file_id))
            self.image1_list.append(x[0])
            self.image2_list.append(x[2])
            self.calib_list.append(x[-1])

            self.disp1_ms_list.append(f'./data_scene_flow/training/disp_msnet{msnet_mode}/{file_id}_10.png')
            self.disp2_ms_list.append(f'./data_scene_flow/training/disp_msnet{msnet_mode}/{file_id}_11.png')

    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, index):
        intrinsics = self.intrinsics_list[index]
        image1 = cv2.imread(osp.join(self.root, self.image1_list[index]))
        image2 = cv2.imread(osp.join(self.root, self.image2_list[index]))

        disp1 = cv2.imread(osp.join(self.root, self.disp1_ms_list[index]), cv2.IMREAD_ANYDEPTH) / 256.0
        disp2 = cv2.imread(osp.join(self.root, self.disp2_ms_list[index]), cv2.IMREAD_ANYDEPTH) / 256.0

        image1 = image1[self.crop:]
        image2 = image2[self.crop:]
        disp1 = disp1[self.crop:]
        disp2 = disp2[self.crop:]
        intrinsics[3] -= self.crop

        image1 = torch.from_numpy(image1).float().permute(2, 0, 1)
        image2 = torch.from_numpy(image2).float().permute(2, 0, 1)
        disp1 = torch.from_numpy(disp1).float()
        disp2 = torch.from_numpy(disp2).float()
        intrinsics = torch.from_numpy(intrinsics).float()

        return self.file_idx[index], image1, image2, disp1, disp2, intrinsics
