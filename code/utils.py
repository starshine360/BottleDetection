import os
import cv2

import torch
import torch.nn as nn
from torchvision import models


def store_components_of_resnet():
    ''' Store some components of a pretrained resnet-50.
    '''
    directory = 'model/component/'

    model = models.resnet50(pretrained=True)

    layer0 = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool
    )

    torch.save(layer0, directory + 'resnet_layer0.pth')
    torch.save(model.layer1, directory + 'resnet_layer1.pth')
    torch.save(model.layer2, directory + 'resnet_layer2.pth')
    torch.save(model.layer3, directory + 'resnet_layer3.pth')
    torch.save(model.layer4, directory + 'resnet_layer4.pth')


def draw_bbox_of_image(img_dir, gt_dir):
    ''' Draw bounding box in Image.
    '''
    img_name_list = os.listdir(img_dir)
    img_name_list.sort()
    gt_name_list = os.listdir(gt_dir)
    gt_name_list.sort()
    
    assert len(img_name_list) == len(gt_name_list)

    for (img_name, gt_name) in zip(img_name_list, gt_name_list):
        assert img_name[:-4] == gt_name[:-4]
        img = cv2.imread(img_dir + img_name)
        with open(gt_dir + gt_name, mode='r') as f:
            line = f.readline()
            gt = line.strip().split(' ')
            [x1, y1, x2, y2] = [int(o) for o in gt]
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=3)
        cv2.imwrite('data/temp/' + img_name, img)

