import os
from plyfile import PlyData
import numpy as np

from torch.utils.data import Dataset

from .templates import text_prompts, omni3d_gpt35_prompts


def normalize_pc(pc):
    # normalize pc to [-1, 1]
    pc = pc - np.mean(pc, axis=0)
    if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc


class OmniObject3D(Dataset):

    def __init__(self, cfg):
        self.lm3d = cfg.lm3d
        
        # option 1: use the manual template from `templates.py`
        self.template = text_prompts
        # option 2: use the responses from the LLM
        # self.template = omni3d_gpt35_prompts

        self.dataset_dir = cfg.omniobject3d_root
        self.num_points = cfg.npoints

        self.classnames = self.set_classnames()
        
        self.test_data, self.test_label = self.load_data()

    def set_classnames(self):
        classnames = []
        print('===', f'{self.dataset_dir}/{self.num_points}', '===')
        for cls in os.listdir(f'{self.dataset_dir}/{self.num_points}'):
            if os.path.isdir(os.path.join(f'{self.dataset_dir}/{self.num_points}', cls)):
                classnames.append(cls)
            
        return sorted(classnames)

    def load_data(self):
        all_data = []
        all_label = []

        data_dir1 = f'{self.dataset_dir}/{self.num_points}'
        for cls in os.listdir(data_dir1):
            data_dir2 = os.path.join(data_dir1, cls)
            for ins in os.listdir(data_dir2):
                data_dir3 = os.path.join(data_dir2, ins)
                if not os.listdir(data_dir3):   # empty dir
                    continue

                data_f = os.path.join(data_dir3, f'pcd_{self.num_points}.ply')
                plydata = PlyData.read(data_f)
                x = plydata.elements[0].data['x']
                y = plydata.elements[0].data['y']
                z = plydata.elements[0].data['z']
                # a whole point cloud
                pts = np.stack([x,y,z], axis=0).T
                # pc's label
                label = self.classnames.index(cls)
                all_data.append(pts)
                all_label.append(label)

        all_data = np.array(all_data)
        all_label = np.array(all_label)
        
        return all_data, all_label

    def __len__(self):
        return len(self.test_label)
    
    def __getitem__(self, idx):
        pc = self.test_data[idx].astype(np.float32)
        
        # NOTE swap y,z axises
        if self.lm3d == 'openshape':
            pc[:, [1, 2]] = pc[:, [2, 1]]
        
        # NOTE it's necessary for `omin3d` to normalize the pc for performance
        pc = normalize_pc(pc)

        label = self.test_label[idx].astype(np.int32)
        classname = self.classnames[int(label)]
        
        rgb = np.ones_like(pc) * 0.4
        
        return pc, label, classname, rgb
