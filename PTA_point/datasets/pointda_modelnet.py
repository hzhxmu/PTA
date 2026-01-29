import os
import numpy as np

from torch.utils.data import Dataset

from .templates import text_prompts
from .utils import normalize_pc, pc_normalize


class PointDA_ModelNet(Dataset):
    def __init__(self, cfg):
        self.lm3d = cfg.lm3d
        self.template = text_prompts
        
        self.dataset_dir = os.path.join('data/xset/pointda', 'modelnet')
        self.classnames, name2idx = self.read_classnames(self.dataset_dir)

        self.test_data, self.test_label = self.load_data(name2idx, 'test')

    def load_data(self, name2idx, split):
        data_list, label_list = [], []
        for cls in os.listdir(self.dataset_dir):
            cls_dir = os.path.join(self.dataset_dir, cls)   # data/pointda/modelnet/bed
            
            if os.path.isdir(cls_dir):
                dir_f = os.path.join(cls_dir, split)    # data/pointda/modelnet/bed/train
                label = name2idx[cls]

                for f in os.listdir(dir_f):
                    if f.endswith('.npy'):
                        # shape: (2048, 3) -> (1, 2048, 3)
                        points = np.expand_dims(np.load(os.path.join(dir_f, f)), axis=0)
                        data_list.append(points)
                        label_list.append([label])

        data = np.concatenate(data_list, axis=0).astype('float32')
        label = np.array(label_list).astype("int64")

        return data, label

    @staticmethod
    def read_classnames(dataset_dir):
        classnames = []
        name2idx = dict()

        names = sorted(os.listdir(dataset_dir))

        for idx, name in enumerate(names):
            if os.path.isdir(os.path.join(dataset_dir, name)):
                classnames.append(name)
                name2idx[name] = idx
        
        return classnames, name2idx

    def __len__(self):
        return len(self.test_label)
    
    def __getitem__(self, idx):
        xyz = self.test_data[idx]
        label = self.test_label[idx]
        cname = self.classnames[int(label)]
        
        if self.lm3d == 'openshape':
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
            xyz = normalize_pc(xyz)
        else:
            xyz = pc_normalize(xyz)
        
        rgb = np.ones_like(xyz) * 0.4
        
        return xyz, label, cname, rgb