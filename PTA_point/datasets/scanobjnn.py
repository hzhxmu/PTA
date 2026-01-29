import copy
import numpy as np

import torch
from torch.utils.data import Dataset

from .templates import text_prompts, sonn_gpt35_prompts, sonn_gpt4_prompts, sonn_pointllm_prompts


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


class ScanObjNN(Dataset):
    def __init__(self, config):
        self.lm3d = config.lm3d
        
        # option 1: use the manual template from `templates.py`
        self.template = text_prompts
        # option 2: use the responses from the LLM
        # self.template = sonn_gpt35_prompts
        # self.template = sonn_gpt4_prompts
        # self.template = sonn_pointllm_prompts

        self.npoints = config.npoints
        self.data_path = config.scanobjnn_root

        self.classnames = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", 
                           "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet"]

        self.openshape_data = np.load('%s/xyz_label.npy' % self.data_path, allow_pickle=True).item()
        
    def __len__(self):
        return len(self.openshape_data['xyz'])
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, index):
        pc = copy.deepcopy(self.openshape_data['xyz'][index][:self.npoints])

        xyz = pc

        if 'rgb' not in self.openshape_data:
            rgb = np.ones_like(xyz) * 0.4
        else:
            rgb = self.openshape_data['rgb'][index]

        if self.lm3d == 'openshape':
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
            xyz = normalize_pc(xyz)
        else:
            xyz = pc_normalize(xyz)

        xyz = torch.from_numpy(xyz).float()
        rgb = torch.from_numpy(rgb).float()

        label = self.openshape_data['label'][index]
        label_name = self.classnames[label]
        label = label.astype(np.int32)

        return xyz, label, label_name, rgb