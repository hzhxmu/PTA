import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .templates import text_prompts, text_prompts_pc2_view


class ModelNet_C_Views(Dataset):
    """ModelNet_C(orruption).

    This dataset is used for testing only.
    """

    def __init__(self, cfg):
        # self.template = text_prompts
        self.template = text_prompts_pc2_view
        
        self.dataset_dir = cfg.modelnet_c_views_root
        self.cor_type = cfg.cor_type

        self.classnames = []
        text_file = os.path.join(self.dataset_dir, 'shape_names.txt')
        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                classname = line.strip()
                self.classnames.append(classname)

        self.pc_view_path = []
        pc_id = 0
        for item in os.listdir(os.path.join(self.dataset_dir, self.cor_type)):
            d = os.path.join(self.dataset_dir, self.cor_type, item)
            if os.path.isdir(d):
                self.pc_view_path.append(os.path.join(self.dataset_dir, self.cor_type, str(pc_id)))
                pc_id += 1

        self.labels = []
        with open(os.path.join(self.dataset_dir, self.cor_type, 'labels.txt')) as fin:
            lines = fin.readlines()
            for line in lines:
                self.labels.append(int(line.strip()))

        # --- option 1. try simplest data augmentation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

        # --- option 2. data augmentation provided by OpenAI `CLIP` or `open_clip`
        # self.transform = preprocess_val

    def __len__(self):
        return len(self.pc_view_path)

    def __getitem__(self, idx):
        pc_view_dir = self.pc_view_path[idx]

        imgs = []
        for view in os.listdir(pc_view_dir):
            img = Image.open(os.path.join(pc_view_dir, view)).convert('RGB')
            if self.transform:
                # img: (3, 224, 224)
                img = self.transform(img)
            imgs.append(img)

        label = self.labels[idx]

        return torch.stack(imgs), label