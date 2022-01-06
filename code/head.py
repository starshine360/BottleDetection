import math

import torch
import torch.nn as nn


class ScaleExp(nn.Module):
    ''' 
    Define a trainable scalar s_i to automatically adjust the base 
    of the exponential function for feature level Pi.
    '''
    def __init__(self, init_value=1.0):
        ''' Initialize
        '''
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float))
    

    def forward(self, X):
        ''' Operations.
        '''
        return torch.exp(X * self.scale)
    

class Head(nn.Module):
    ''' Define the head part of the network.
    '''
    def __init__(self, in_channel=256):
        ''' Initialize.
        '''
        super(Head, self).__init__()

        self.cls_conv = nn.Sequential( 
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1), 
            nn.GroupNorm(32, in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1), 
            nn.GroupNorm(32, in_channel),
            nn.ReLU(inplace=True)
        )

        self.reg_conv = nn.Sequential( 
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1), 
            nn.GroupNorm(32, in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1), 
            nn.GroupNorm(32, in_channel),
            nn.ReLU(inplace=True)
        )

        self.cls_end = nn.Conv2d(in_channel, 1, kernel_size=3, padding=1)    # classification
        self.ctn_end = nn.Conv2d(in_channel, 1, kernel_size=3, padding=1)    # center-ness
        self.reg_end = nn.Conv2d(in_channel, 4, kernel_size=3, padding=1)    # regression

        self.apply(self._init_conv_param)

        # self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(N_Feature_Map)])


    def forward(self, X):
        ''' Define operations.

        Arguments:
            X: P -- [N, C, H, W]
        
        Returns:
            cls_out: [N, H, W]
            ctn_out: [N, H, W]
            reg_out: [N, H, W, 4]
        '''
        cls_conv_out = self.cls_conv(X)
        reg_conv_out = self.reg_conv(X)

        cls_out = self.cls_end(cls_conv_out).squeeze(dim=1)
        cls_out = torch.sigmoid(cls_out)
        
        ctn_out = self.ctn_end(reg_conv_out).squeeze(dim=1)
        ctn_out = torch.sigmoid(ctn_out)
            
        reg_out = torch.exp(self.reg_end(reg_conv_out))
        reg_out = reg_out.permute(0, 2, 3, 1)

        return cls_out, ctn_out, reg_out
    
    
    def _init_conv_param(self, module, std=0.01):
        ''' Initialize the parameters of Conv2d.
        '''
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)