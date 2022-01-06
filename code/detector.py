from types import CellType
import cv2
import numpy as np
import torch
from torchvision.io import read_image

from .config import Min_Conf, Max_Num
from .config import Feature_Map_Size, Scale, Image_Size


class Detector(object):
    ''' Bottle Detector.
    '''
    def __init__(self, model_path: str):
        ''' Initialization.
        '''
        super(Detector, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.load(model_path).to(self.device)
        self.model.eval()


    def detect(self, img_path):
        ''' Detect area of the bottle.

        Return:
            bbox: (x1, y1, x2, y2) or None
        '''
        # Load image
        img = read_image(img_path).float()
        img = img.unsqueeze(dim=0)    # [1, C, H, W]
        img = img.to(self.device)

        # Prediction
        cls_out, ctn_out, reg_out = self.model(img)
        cls_out = cls_out.squeeze()
        ctn_out = ctn_out.squeeze()
        reg_out = reg_out.squeeze()
        
        # Screen
        candidate_list = []
        H, W = Feature_Map_Size
        for i in range(W):
            for j in range(H):
                x = i * Scale + Scale // 2
                y = j * Scale + Scale // 2

                cls = cls_out[j, i].item()
                ctn = ctn_out[j, i].item()
                l, r, t, b = reg_out[j, i, :].tolist()

                group = (x, y, l, r, t, b)
                bbox = self._transform(group)

                score = cls * ctn
                candidate = (score, ) + bbox

                if cls > Min_Conf:
                    candidate_list.append(candidate)
        
        # Sort by score and select the top k
        candidate_list = sorted(candidate_list, key=lambda c: c[0], reverse=True)
        if len(candidate_list) > Max_Num:
            candidate_list = candidate_list[0:Max_Num]
        
        if len(candidate_list) > 0:
            # Compute mean value with the weight of score
            score_sum = sum([c[0] for c in candidate_list])
            avg_x1 = sum([c[0] * c[1] for c in candidate_list]) / score_sum
            avg_y1 = sum([c[0] * c[2] for c in candidate_list]) / score_sum
            avg_x2 = sum([c[0] * c[3] for c in candidate_list]) / score_sum
            avg_y2 = sum([c[0] * c[4] for c in candidate_list]) / score_sum

            bbox = (int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2))
        else:
            bbox = None
        
        return bbox


    def locate(self, img_path):
        ''' Locate the bottleneck.
        '''
        img = cv2.imread(img_path)

        # 高斯模糊
        # gaussian_img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # 灰度图像
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Hough检测圆
        H, W = gray_img.shape
        circles = cv2.HoughCircles(gray_img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=H, param1=100, param2=80, minRadius=5, maxRadius=50)
        circles = np.uint16(np.around(circles.reshape(-1, 3)))

        # 绘制圆
        for c in circles:
            cv2.circle(img, center=(c[0], c[1]), radius=c[2], color=(0, 0, 255), thickness=3)    # 圆
            cv2.circle(img, center=(c[0], c[1]), radius=2, color=(255, 0, 0), thickness=3)    # 圆心
        cv2.imshow('result', img)
        cv2.waitKey()


    def _transform(self, group):
        ''' Transform (x, y, l, r, t, b) to (x1, y1, x2, y2)
        '''
        H, W = Image_Size
        x, y, l, r, t, b = group

        x1 = max(x - l, 0)
        y1 = max(y - t, 0)
        x2 = min(x + r, W)
        y2 = min(y + b, H)
        
        return (x1, y1, x2, y2)

    
    def _stretch(self, img):
        ''' 进行灰度变换
        '''
        min_val = img.min()
        max_val = img.max()

        img = (img - min_val) * (255 / (max_val - min_val))
        img = img.astype(np.uint8)

        return img 
