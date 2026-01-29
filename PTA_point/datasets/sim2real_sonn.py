import os
import numpy as np

from torch.utils.data import Dataset

from .templates import text_prompts


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


class Sim2Real_SONN(Dataset):

    def __init__(self, cfg):
        """
        This dataset is used for testing only.
        """
        self.lm3d = cfg.lm3d
        self.template = text_prompts
        
        self.npoints = cfg.npoints  # 2048 by default
        sim2real_type = cfg.sim2real_type
        self.dataset_dir = os.path.join('data/xset/sim2real', sim2real_type)

        self.classnames, name2idx = self.read_classnames(self.dataset_dir)
        
        self.test_data, self.test_label = self.load_data(name2idx, 'test')

    def load_data(self, name2idx, split):
        data_list, label_list = [], []
        for cls in os.listdir(self.dataset_dir):
            cls_dir = os.path.join(self.dataset_dir, cls)   # data/xset/sim2real/so_obj_only_9/bed
            
            if os.path.isdir(cls_dir):
                dir_f = os.path.join(cls_dir, split)    # data/xset/sim2real/shapenet_9/bed/test
                label = name2idx[cls]

                for f in os.listdir(dir_f):
                    # shape: (2048, 3) -> (1, 2048, 3)
                    points = np.expand_dims(np.load(os.path.join(dir_f, f)), axis=0)
                    data_list.append(points)
                    label_list.append([label])

        data = np.concatenate(data_list, axis=0).astype('float32')
        label = np.array(label_list).astype("int64")

        return data, label

    @staticmethod
    def read_classnames(dataset_dir):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
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
        xyz = self.test_data[idx][:self.npoints]
        label = self.test_label[idx]
        cname = self.classnames[int(label)]
        
        if self.lm3d == 'openshape':
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
            xyz = normalize_pc(xyz)
        else:
            xyz = pc_normalize(xyz)
        
        rgb = np.ones_like(xyz) * 0.4
        
        return xyz, label, cname, rgb
