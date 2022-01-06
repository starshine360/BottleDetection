import torch
import torch.nn as nn


class Backbone(nn.Module):
    ''' Use components of a pretrained resnet50 model as the backbone.
    '''
    def __init__(self):
        ''' Initialize the network.
        '''
        super(Backbone, self).__init__()

        # Load components of resnet50
        directory = 'model/component/'
        self.layer0 = self._load_component(directory + 'resnet_layer0.pth')
        self.layer1 = self._load_component(directory + 'resnet_layer1.pth')
        self.layer2 = self._load_component(directory + 'resnet_layer2.pth')    # out_channels is 512
        self.layer3 = self._load_component(directory + 'resnet_layer3.pth')    # out_channels is 1024
        self.layer4 = self._load_component(directory + 'resnet_layer4.pth')    # out_channels is 2048

    
    def forward(self, X):
        ''' Define operations for input data.

        Arguments:
            X: Image Set - tensor of [N, C, H, W]
        
        Return:
            C: tensor of [N, 2048, H / 32, W / 32]
        '''
        X = self.layer0(X)
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        C = self.layer4(X)

        return C


    def _load_component(self, path):
        ''' Load components of a pretrained resnet50 model.
        '''
        component = torch.load(path)
    
        for param in component.parameters():
            param.requires_grad = False

        return component

