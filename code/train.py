import random
import gc

import torch
import torch.nn.functional as F
from torch.optim import optimizer

from .net import myNet
from .data import load_data
from .config import Feature_Map_Size, Scale


def generate_targets(bbox):
    ''' Generate targets for a sample.

    Returns:
        mask: Tensor - [H, W]
        cls_tag: Tensor - [H, W]
        ctn_tag: Tensor - [H, W]
        reg_tag: Tensor - [H, W, 4]
    '''
    x1, y1, x2, y2 = bbox

    H, W = Feature_Map_Size
    x_vec = torch.arange(0, W * Scale, Scale)
    y_vec = torch.arange(0, H * Scale, Scale)
    y_matrix, x_matrix = torch.meshgrid(y_vec, x_vec)
    x_matrix = x_matrix + Scale // 2
    y_matrix = y_matrix + Scale // 2
   
    mask = (x1 < x_matrix) & (x_matrix < x2) & (y1 < y_matrix) & (y_matrix < y2)
    
    cls_tag = mask.float()
    
    left = (x_matrix - x1) * mask
    right = (x2 - x_matrix) * mask
    top = (y_matrix - y1) * mask
    buttom = (y2 - y_matrix) * mask

    reg_tag = torch.stack((left, right, top, buttom), dim=2)   
    
    ctn_tag = torch.sqrt( (torch.min(left, right) * torch.min(top, buttom)) / (torch.max(left, right) * torch.max(top, buttom) + 1e-10) )

    return mask, cls_tag, ctn_tag, reg_tag


def compute_cls_loss(predict, target):
    ''' Compute classification loss, using focal loss.

    Argument:
        mask: Tensor - [H, W]
        predict: Tensor - [H, W]
        target: Tensor - [H, W]
    '''
    h, w = predict.shape
    
    # loss = F.binary_cross_entropy(predict, target)
    gamma = 2
    pt = predict * target + (1.0 - predict) * (1.0 - target)
    pt = pt.clamp(min=1e-6)
    loss = - torch.pow(1.0 - pt, gamma) * torch.log(pt)

    return torch.sum(loss) / (h * w)


def compute_ctn_loss(mask, predict, target):
    ''' Compute center-nesss loss.

    Argument:
        mask: Tensor - [H, W]
        predict: Tensor - [H, W]
        target: Tensor - [H, W]
    '''
    num_pos = torch.sum(mask)
    
    mask_pred = torch.masked_select(predict, mask)
    mask_tag = torch.masked_select(target, mask)
    
    loss = F.binary_cross_entropy(mask_pred, mask_tag)
    
    return torch.sum(loss) / num_pos


def compute_reg_loss(mask, predict, target):
    ''' Compute regression loss, using IoU loss

    Arguments:
        mask: [H, W]
        predict: [H, W, 4]
        target: [H, W, 4]
    ''' 
    num_pos = torch.sum(mask)

    mask_pred = predict[mask]
    mask_tag = target[mask]
    
    area1 = (mask_tag[:, 0] + mask_tag[:, 1]) * (mask_tag[:, 2] + mask_tag[:, 3])
    area2 = (mask_pred[:, 0] + mask_pred[:, 1]) * (mask_pred[:, 2] + mask_pred[:, 3])
    
    min_val = torch.min(mask_pred, mask_tag)
    overlap = (min_val[:, 0] + min_val[:, 1]) * (min_val[:, 2] + min_val[:, 3])
    
    iou = overlap / (area1 + area2 - overlap)
    iou = iou.clamp(min=1e-6)
    
    loss = - torch.log(iou)
    
    return torch.sum(loss) / num_pos


def compute_loss(cls_pred, ctn_pred, reg_pred, bbox):
    ''' Compute loss for a single sample.

    Arguments:
        cls_pred: Tensor - [H, W]
        ctn_pred: Tensor - [H, W]
        reg_pred: Tensor - [H, W, 4]
    '''
    mask, cls_tag, ctn_tag, reg_tag = generate_targets(bbox)
    
    cls_loss = compute_cls_loss(cls_pred, cls_tag)
    ctn_loss = compute_ctn_loss(mask, ctn_pred, ctn_tag)
    reg_loss = compute_reg_loss(mask, reg_pred, reg_tag)
    
    return cls_loss + ctn_loss + reg_loss



def train(choice: int = 1, max_epoch: int = 50, batch_size: int = 8, lr: float = 0.0001):
    ''' Train the model.
    '''
    assert choice >= 0 and choice <= 3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_dir = 'data/type{}/image/'.format(choice)
    gt_dir = 'data/type{}/groundtruth/'.format(choice)

    img_set, bbox_set = load_data(img_dir, gt_dir)
    
    model = myNet().to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.7, weight_decay=0.001)
    
    N, C, H, W = img_set.shape
    idx_list = [i for i in range(N)]

    # Set state of model
    model.train()
    model.backbone.eval()
    
    for epoch in range(max_epoch):
        print('At epoch {}'.format(epoch))

        # shuffle
        random.shuffle(idx_list)

        # Set learning rate
        # if epoch == max_epoch // 2:
        #     for param in optimizer.param_groups:
        #         param['lr'] = 0.0005
        
        # Begin to train
        base = 0
        while base + batch_size <= N:
            # load data
            batch_img_set = img_set[idx_list[base : base+batch_size]]
            batch_bbox_set = bbox_set[idx_list[base : base+batch_size]]
            
            # Compute prediction
            cls_out, ctn_out, reg_out = model(batch_img_set.to(device))
            cls_out = cls_out.to('cpu')
            ctn_out = ctn_out.to('cpu')
            reg_out = reg_out.to('cpu')
    
            reg_out.retain_grad()
            
            # Compute loss
            cls_loss = torch.tensor(0).float()
            ctn_loss = torch.tensor(0).float()
            reg_loss = torch.tensor(0).float()
            for idx in range(batch_size):
                mask, cls_tag, ctn_tag, reg_tag = generate_targets(batch_bbox_set[idx])
                cls_loss += compute_cls_loss(cls_out[idx], cls_tag)
                ctn_loss += compute_ctn_loss(mask, ctn_out[idx], ctn_tag)
                reg_loss += compute_reg_loss(mask, reg_out[idx], reg_tag)
            cls_loss = cls_loss / batch_size
            ctn_loss = ctn_loss / batch_size
            reg_loss = reg_loss / batch_size
            print('        cls_loss is {:.3f}, ctn_loss is {:.3f}, reg_loss is {:.3f}'.format(cls_loss.item(), ctn_loss.item(), reg_loss.item()))
            loss = cls_loss + ctn_loss + reg_loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()

            # update
            base += batch_size
        
        # gc.collect()
        # torch.cuda.empty_cache()

    # save
    model.to('cpu')
    torch.save(model, 'model/myNet_type{}.pth'.format(choice))




"""    
torch.nn.utils.clip_grad_norm_(parameters=model.middle.parameters(), max_norm=3, norm_type=2, error_if_nonfinite=True)
            
"""