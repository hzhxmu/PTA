import os
import numpy as np

from torch.utils.data import Dataset

from .templates import text_prompts


class ModelNet40_C(Dataset):
    """ModelNer40-C(orruption).

    This dataset is used for testing only.
    """

    def __init__(self, cfg):
        self.template = text_prompts

        self.dataset_dir = cfg.modelnet40_c_root

        self.classnames = []
        text_file = os.path.join(self.dataset_dir, 'shape_names.txt')
        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                classname = line.strip()
                self.classnames.append(classname)

        cor_type = cfg.cor_type
        data_file = f'data_{cor_type}.npy'  # e.g., background_1, cutout_1, density_1

        self.test_data = np.load(f'{self.dataset_dir}/{data_file}')
        self.test_label = np.load(f'{self.dataset_dir}/label.npy')

        self.npoints = cfg.npoints
    
    def __len__(self):
        return len(self.test_label)

    def __getitem__(self, idx):
        pc = self.test_data[idx].astype(np.float32)
        label = self.test_label[idx].astype(np.int32)
        classname = self.classnames[int(label)]
        
        rgb = np.ones_like(pc) * 0.4
        return pc, label, classname, rgb
