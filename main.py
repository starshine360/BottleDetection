import cv2
import os

from code.utils import store_components_of_resnet
from code.train import train
from code.detector import Detector


Choice = 3


if Choice == 0:
    #####################
    ## Prepare
    #####################

    store_components_of_resnet()


elif Choice == 1:
    ####################
    # Train
    ####################

    train(choice=1, max_epoch=100, batch_size=16, lr=0.0001)


elif Choice == 2:
    ########################
    ## Evaluate Detection
    ########################

    Bottom_Type = 3

    model_path = 'model/myNet_type{}.pth'.format(Bottom_Type)
    myDetector = Detector(model_path)

    img_dir = 'data/type{}/image/'.format(Bottom_Type)
    gt_dir = 'data/type{}/groundtruth/'.format(Bottom_Type)
    img_name_list = os.listdir(img_dir)
    img_name_list.sort()
    gt_name_list = os.listdir(gt_dir)
    gt_name_list.sort()

    bbox_set = []
    for gt_name in gt_name_list:
        gt_path = gt_dir + gt_name
        with open(gt_path, mode='r') as f:
            line = f.readline()
            bbox = line.strip().split(' ')
            bbox = [int(o) for o in bbox]
            bbox_set.append(bbox)

    p_bbox_set = []
    for img_name in img_name_list:
        img_path = img_dir + img_name
        bbox = myDetector.detect(img_path)
        if not bbox:
            raise Exception('Find no bbox for {}'.format(img_name))
        p_bbox_set.append(bbox)
    
    iou_set = []
    for bbox, p_bbox in zip(bbox_set, p_bbox_set):
        x1, y1, x2, y2 = bbox
        p_x1, p_y1, p_x2, p_y2 = p_bbox
        m_x1, m_y1, m_x2, m_y2 = max(x1, p_x1), max(y1, p_y1),  min(x2, p_x2), min(y2, p_y2)
        
        if m_x1 < m_x2 and m_y1 < m_y2:
            overlap = (m_x2 - m_x1) * (m_y2 - m_y1)
        else:
            overlap = 0
        
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (p_x2 - p_x1) * (p_y2 - p_y1)
        
        iou = overlap / (area1 + area2 - overlap)
        iou_set.append(iou)
    
    print('  max iou of type {} is {}'.format(Bottom_Type, max(iou_set)))
    print('  min iou of type {} is {}'.format(Bottom_Type, min(iou_set)))
    print('  mean iou of type {} is {}'.format(Bottom_Type, sum(iou_set) / len(iou_set)))


elif Choice == 3:
    #####################
    ## Test Detection
    #####################

    model_path = 'model/myNet_type1.pth'
    img_path = 'data/type1/image/1_68.jpg'

    myDetector = Detector(model_path)
    bbox = myDetector.detect(img_path)

    img = cv2.imread(img_path)
    if bbox:
        p1 = (bbox[0], bbox[1])
        p2 = (bbox[2], bbox[3])
        cv2.rectangle(img, p1, p2, color=(0, 255, 0), thickness=2)

    cv2.imshow('result', img)
    cv2.waitKey()


elif Choice == 4:
    ########################
    ## Test location
    ########################

    model_path = 'model/myNet_type1.pth'
    img_path = 'data/type0/image/3_17.jpg'

    myDetector = Detector(model_path)

    myDetector.locate(img_path)