import os

import torch
from torchvision.io import read_image


def load_data(img_dir, gt_dir):
    ''' Load images and groundtruth.

    Returns:
        img_set: Tensor - [N, C, H, W]
        bbox_set: Tensor - [N, 4]
    '''
    img_name_list = os.listdir(img_dir)
    img_name_list.sort()
    gt_name_list = os.listdir(gt_dir)
    gt_name_list.sort()

    assert len(img_name_list) == len(gt_name_list)

    img_list = []
    bbox_list = []

    for (img_name, gt_name) in zip(img_name_list, gt_name_list):
        # check
        assert img_name[:-4] == gt_name[:-4]
        # load image
        img_path = img_dir + img_name
        img = read_image(img_path).float()
        img_list.append(img)
        # load 
        gt_path = gt_dir + gt_name
        with open(gt_path, mode='r') as f:
            line = f.readline()
            bbox = line.strip().split(' ')
            bbox = [int(o) for o in bbox]
            bbox = torch.tensor(bbox)
            bbox_list.append(bbox)
        
    img_set = torch.stack(img_list, dim=0)
    bbox_set = torch.stack(bbox_list, dim=0)

    return img_set, bbox_set 


