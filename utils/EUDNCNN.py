#%%
from structures import *
import torch
from torch import nn
import copy
#%%
'''model define'''
class EUDnCNN(nn.Module):
    def __init__(self, D, C = 32):
        super(EUDnCNN, self).__init__()
        self.D = D
        self.C = C
        # convolution layers
        self.conv = nn.ModuleList()
        # self.conv.append(nn.Conv2d(1, C, 3, padding=1))
        self.conv.append(SobelConv2d(1, 2*C, 3, padding=1))
        self.conv.extend([nn.Conv2d(2*C+1, C, 3, padding=1) for _ in range(D)])
        self.conv.append(nn.Conv2d(2*C+1, 1, 3, padding=1))
        # apply He's initialization
        for i in range(1, len(self.conv[:-1])):
            nn.init.kaiming_normal_(self.conv[i].weight.data, nonlinearity='relu')
        
        # batch normalization
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
        # initialize the weights of the Batch normalization layers
        for i in range(D):
            nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))
        
        self.conv_sobel = SobelConv2d(1, C, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x): 
        
        D = self.D
        # h = F.relu(self.conv[0](x))
        h = self.conv[0](x)
        h = torch.cat((x, h), dim = -3)
        '''get shapes'''
        h_buff = []
        idx_buff = []
        shape_buff = []
        for i in range(D//2-1):
            shape_buff.append(h.shape)
            h, idx = F.max_pool2d(F.relu(self.bn[i](self.conv[i+1](h))), 
                                  kernel_size=(2,2), return_indices=True)
            input_im = copy.deepcopy(x)
            input_im.resize_((h.shape[0], 1, h.shape[2], h.shape[3]))
            sobeled = self.conv_sobel(input_im)
            sobel_op = torch.cat((input_im, sobeled), dim = -3)
            h = torch.cat((sobel_op, h), dim = -3)

            h_buff.append(h)
            idx_buff.append(idx)

        # return h_buff, idx_buff
        for i in range(D//2-1, D//2+1):
            h = F.relu(self.bn[i](self.conv[i+1](h)))

            input_im = copy.deepcopy(x)
            input_im.resize_((h.shape[0],1, h.shape[2], h.shape[3]))
            sobeled = self.conv_sobel(input_im)
            sobel_op = torch.cat((input_im, sobeled), dim = -3)
            
            h = torch.cat((sobel_op, h), dim = -3)

        for i in range(D//2+1, D):
            j = i - (D//2 + 1) + 1
            h = F.max_unpool2d(F.relu(self.bn[i](self.conv[i+1]((h+h_buff[-j])/np.sqrt(2)))), 
                               idx_buff[-j], kernel_size=(2,2), output_size=shape_buff[-j])
            
            input_im = copy.deepcopy(x)
            input_im.resize_((h.shape[0],1, h.shape[2], h.shape[3]))
            sobeled = self.conv_sobel(input_im)
            sobel_op = torch.cat((input_im, sobeled), dim = -3)
            
            h = torch.cat((sobel_op, h), dim = -3)
        y = self.conv[D+1](h) + x
        return y