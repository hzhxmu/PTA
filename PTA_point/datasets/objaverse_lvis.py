import os
import numpy as np

import torch
from torch.utils.data import Dataset

# NOTE *** Here I just use the default templates from ULIP, it may need to customize for each dataset
from .templates import text_prompts, o_lvis_gpt35_prompts


def pc_normalize(pc):
    ''' NOTE what's the difference between `pc_normalize` and `normalize_pc`??? '''
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def normalize_pc(pc):
    # normalize pc to [-1, 1]
    pc = pc - np.mean(pc, axis=0)
    if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


class Objaverse_LVIS(Dataset):
    def __init__(self, config):
        # NOTE how to replace `config`
        
        self.lm3d = config.lm3d   # False by default
        
        self.npoints = config.npoints

        self.pc_root = config.objaverse_lvis_root
        self.pc_list = os.path.join(self.pc_root, 'lvis_testset.txt')

        self.file_list = []
        self.classnames = []
        with open(self.pc_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                clsname = line.split(',')[1]
                self.file_list.append({
                    'cate_id': line.split(',')[0],
                    'cate_name': clsname,
                    'model_id': line.split(',')[2],
                    'point_path': self.pc_root + line.split(',')[3].replace('\n', '')
                })
                if clsname not in self.classnames:
                    self.classnames.append(clsname)

        # option 1: use the manual template from `templates.py`
        self.template = text_prompts
        # option 2: use the responses from the LLM
        # self.template = o_lvis_gpt35_prompts

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        cate_id = sample['cate_id']
        cate_name = sample['cate_name']
        point_path = sample['point_path']

        while True:
            try:
                openshape_data = np.load(point_path, allow_pickle=True).item()
                data = openshape_data['xyz'][:self.npoints].astype(np.float32)
                rgb = openshape_data['rgb'][:self.npoints].astype(np.float32)
            except OSError as e:
                print(f"Catched exception: {str(e)}. Re-trying...")
                import time
                time.sleep(1)
            else:
                break

        if self.lm3d == 'openshape':
            data[:, [1, 2]] = data[:, [2, 1]]
            data = normalize_pc(data)
        else:
            data = pc_normalize(data)
            
        data = torch.from_numpy(data).float()

        cate_id = np.array([cate_id]).astype(np.int32)
        return data, cate_id, cate_name, rgb
