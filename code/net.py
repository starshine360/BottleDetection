import torch.nn as nn

from .backbone import Backbone
from .middle import Middle
from .head import Head


class myNet(nn.Module):
    ''' Define the network
    '''
    def __init__(self):
        ''' Initialize.
        '''
        super(myNet, self).__init__()

        self.backbone = Backbone()
        self.middle = Middle(2048, 256)
        self.head = Head(256)
    
    
    def forward(self, X):
        ''' Operations.

        Arguments:
            X: Image Set - [N, C, H, W]

        Returns:
            cls_out: [N, H, W]
            ctn_out: [N, H, W]
            reg_out: [N, H, W, 4]
        '''
        C = self.backbone(X)  
        P = self.middle(C)
        cls_out, ctn_out, reg_out = self.head(P)

        return cls_out, ctn_out, reg_out

        