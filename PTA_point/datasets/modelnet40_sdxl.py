import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .templates import text_prompts, text_prompts_pc2_view


class ModelNet40_SDXL(Dataset):
    """ This dataset is used for testing only. """

    def __init__(self, cfg):
        self.template = text_prompts_pc2_view
        
        self.dataset_dir = cfg.modelnet40_sdxl_root

        self.classnames = []
        text_file = os.path.join(self.dataset_dir, 'shape_names.txt')
        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                classname = line.strip()
                self.classnames.append(classname)
                
        self.pc_img_path = []
        self.labels = []
        for cls in self.classnames:
            d = os.path.join(self.dataset_dir, cls)
            self.pc_img_path.append(d)
            
            label = self.classnames.index(cls)
            self.labels.append(label)
            
        # --- option 1. try simplest data augmentation
        self.transform = transforms.Compose([
            transforms.Resize((cfg.imsize, cfg.imsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

        # --- option 2. data augmentation provided by OpenAI `CLIP` or `open_clip`
        # self.transform = preprocess_val

    def __len__(self):
        return len(self.pc_img_path)

    def __getitem__(self, idx):
        pc_img_dir = self.pc_img_path[idx]

        imgs = []
        for sdxl_img in os.listdir(pc_img_dir):
            img = Image.open(os.path.join(pc_img_dir, sdxl_img)).convert('RGB')
            if self.transform:
                # img: (3, 224, 224)
                img = self.transform(img)
            imgs.append(img)

        label = self.labels[idx]

        return torch.stack(imgs), label