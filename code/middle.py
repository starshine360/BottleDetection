import torch.nn as nn


class Middle(nn.Module):
    ''' Define the structure of middle.
    '''
    def __init__(self, in_channel=2048, out_channel=256):
        ''' Initialize.
        '''
        super(Middle, self).__init__()
        
        self.project = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        self.apply(self._init_conv_param)

    
    def forward(self, X):
        ''' Operations
        '''
        P = self.project(X)
        
        return P


    def _init_conv_param(self, module, std=0.01):
        ''' Initialize the parameters of Conv2d.
        '''
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
