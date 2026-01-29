import os
import h5py
import numpy as np

from torch.utils.data import Dataset

from .templates import text_prompts


class SNV2_C(Dataset):
    """SNV2_C(orruption).

    This dataset is used for testing only.
    """

    def __init__(self, cfg):
        self.lm3d = cfg.lm3d
                
        self.template = text_prompts
        
        self.dataset_dir = cfg.snv2_c_root

        self.classnames = []
        text_file = os.path.join(self.dataset_dir, 'shape_names.txt')
        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                classname = line.strip()
                self.classnames.append(classname)

        cor_type = cfg.cor_type
        data_file = f'{cor_type}.h5'

        f = h5py.File(f'{self.dataset_dir}/{data_file}')
        self.test_data = f['data'][:]
        self.test_label = f['label'][:]

        self.npoints = cfg.npoints

    def __len__(self):
        return len(self.test_label)

    def __getitem__(self, idx):
        pc = self.test_data[idx].astype(np.float32)
        
        # NOTE swap y,z axises
        if self.lm3d == 'openshape':
            pc[:, [1, 2]] = pc[:, [2, 1]]
            
        label = self.test_label[idx].astype(np.int32)
        classname = self.classnames[int(label)]
        
        rgb = np.ones_like(pc) * 0.4
        return pc, label, classname, rgb