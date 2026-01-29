import os
import h5py
import numpy as np

from torch.utils.data import Dataset

from .templates import text_prompts
from .utils import normalize_pc, pc_normalize


class PointDA_ScanNet(Dataset):
    def __init__(self, cfg):
        self.lm3d = cfg.lm3d
        self.template = text_prompts

        self.dataset_dir = os.path.join('data/xset/pointda', 'scannet')

        text_file = os.path.join(self.dataset_dir, 'shape_names.txt')
        self.classnames = self.read_classnames(text_file)

        self.test_data, self.test_label = self.load_data(os.path.join(self.dataset_dir, 'test_files.txt'))

    def load_data(self, data_path):
        all_data = []
        all_label = []
        with open(data_path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(h5_name.strip(), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                all_data.append(data)
                all_label.append(label)
        # NOTE each point has 6 dimensions: first 3 coordinates, laast 3 colors
        all_data = np.concatenate(all_data, axis=0)[:, :, :6]
        all_label = np.concatenate(all_label, axis=0)

        return all_data, all_label

    @staticmethod
    def read_classnames(text_file):
        classnames = []
        with open(text_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                classname = line.strip()
                classnames.append(classname)
                
        return classnames
    
    def __len__(self):
        return len(self.test_label)
    
    def __getitem__(self, idx):
        """ NOTE each point has 6 dimension: xyz, rgb """
        xyz = self.test_data[idx][:, :3]
        rgb = self.test_data[idx][:, 3:]
        
        label = self.test_label[idx]
        cname = self.classnames[int(label)]
        
        if self.lm3d == 'openshape':
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
            xyz = normalize_pc(xyz)
            rgb = normalize_pc(rgb)
        else:
            xyz = pc_normalize(xyz)
            rgb = normalize_pc(rgb)
        
        return xyz, label, cname, rgb
    