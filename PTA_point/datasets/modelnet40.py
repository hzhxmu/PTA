import os
import copy
import json
import numpy as np

import torch
from torch.utils.data import Dataset

from .templates import text_prompts, mn40_gpt35_prompts, mn40_gpt4_prompts, mn40_pointllm_prompts


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


class ModelNet40(Dataset):
    def __init__(self, config):
        self.lm3d = config.lm3d
        # option 1: use the manual template from `templates.py`
        self.template = text_prompts
        # option 2: use the responses from the LLM
        # self.template = mn40_gpt35_prompts
        # self.template = mn40_gpt4_prompts
        # self.template = mn40_pointllm_prompts

        self.npoints = config.npoints
        self.data_path = config.modelnet40_root
        self.catfile = os.path.join(self.data_path, 'classnames.txt')
        self.classnames = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.classnames, range(len(self.classnames))))

        self.pcs = np.load('%s/test_pc.npy' % self.data_path, allow_pickle=True)
        self.openshape_split = json.load(open('%s/test_split.json' % self.data_path, "r"))

        self.cate_to_id = {}
        for i in range(len(self.classnames)):
            self.cate_to_id[self.classnames[i]] = str(i)

    def __len__(self):
        return len(self.openshape_split)

    def __getitem__(self, idx):
        pc = copy.deepcopy(self.pcs[idx])

        xyz = pc['xyz'][:self.npoints]
        rgb = pc['rgb'][:self.npoints]
        rgb = rgb / 255.0 # 100, scale to 0.4 to make it consistent with the training data
        rgb = torch.from_numpy(rgb).float()
        
        # NOTE swap y,z axises
        if self.lm3d == 'openshape':
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
            xyz = normalize_pc(xyz)
        else:
            xyz[:, 0:3] = pc_normalize(xyz[:, 0:3])
        
        xyz = torch.from_numpy(xyz).float()

        label_name = self.openshape_split[idx]["category"]
        label = np.array([int(self.cate_to_id[label_name])]).astype(np.int32)

        return xyz, label[0], label_name, rgb