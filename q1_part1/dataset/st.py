import glob
import os
import random

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import json

class SceneTextDataset(Dataset):
    def __init__(self, split, root_dir):
        self.split = split
        self.root_dir = root_dir
        self.im_dir = os.path.join(root_dir, 'img')
        self.ann_dir = os.path.join(root_dir, 'annots')
        classes = [
            'text'
        ]
        classes = sorted(classes)
        classes = ['background'] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        # print(self.idx2label, "idx2label ") {0: 'background', 1: 'text'}
        self.images = glob.glob(os.path.join(self.im_dir, '*.jpg')) 
        self.annotations = [os.path.join(self.ann_dir, os.path.basename(im) + '.json') for im in self.images]
        
        if(split == 'train'):
            print(f"Images found: {len(self.images)}")

            self.images = self.images[:int(0.85*len(self.images))]
            self.annotations = self.annotations[:int(0.85*len(self.annotations))]

        elif split == 'val':
            self.images = self.images[int(0.85*len(self.images)):int(0.9*len(self.images))]
            self.annotations = self.annotations[int(0.85*len(self.annotations)):int(0.9*len(self.annotations))]
    
        else:
            self.images_test = self.images[int(0.9*len(self.images)):]
            self.annotations_test = self.annotations[int(0.9*len(self.annotations)):]
    
    def __len__(self):
        return len(self.images)
    
    def convert_xcycwh_to_xyxy(self, box):
        x, y, w, h = box
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        return [x1, y1, x2, y2]
    
    def __getitem__(self, index):
        im_path = self.images[index]
        im = Image.open(im_path)
        
        im_tensor = torchvision.transforms.ToTensor()(im)
        targets = {}
        ann_path = self.annotations[index]
        with open(ann_path, 'r') as f:
            im_info = json.load(f)
            xc = [detec['obb']['xc'] for detec in im_info['objects']]
            yc = [detec['obb']['yc'] for detec in im_info['objects']]
            w = [detec['obb']['w'] for detec in im_info['objects']]
            h = [detec['obb']['h'] for detec in im_info['objects']]
            
            # read the angles here as well from the json file...
            
            boxes = [self.convert_xcycwh_to_xyxy([xc[i], yc[i], w[i], h[i]]) for i in range(len(xc))]
            
        targets['bboxes'] = torch.as_tensor(boxes).float()
        targets['labels'] = torch.as_tensor(torch.ones(len(im_info['objects'])).long())
        
        # print(im_tensor.shape,"im_tensor ",index,im_path) #torch.Size([3, 763, 1053]) im_tensor  211 ./../Datasets/Q1/img/img1431.jpg
        # print(targets,"targets ",index,im_path)
        # {'bboxes': tensor([[655.9999, 386.0000, 737.9999, 419.0000],
        # [680.9999, 335.0000, 741.9999, 361.0000],
        # [688.0000, 367.0000, 727.0000, 383.0000],
        # [643.0000, 421.0000, 758.0000, 470.0000],
        # [729.3562, 370.9543, 753.6438, 388.0457],
        # [652.7418, 370.9171, 688.2868, 387.3195],
        # [660.5039, 343.9796, 682.5946, 353.9385]]), 'labels': tensor([1, 1, 1, 1, 1, 1, 1])} targets  211 ./../Datasets/Q1/img/img1431.jpg

        return im_tensor, targets, im_path