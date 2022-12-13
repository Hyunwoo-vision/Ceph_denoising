#%%
'''Packages'''
from fileinput import filename
import os
from unittest import result
from click import Tuple
from cv2 import drawFrameAxes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2
import copy
import math
from torchvision.transforms.functional import to_pil_image
import torch.optim as optim
from torchsummary import summary
from typing import Dict, Optional, List, Tuple

import gc
from skimage.exposure import match_histograms
gc.collect()
torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#########################################################################################################
#%%
'''guassian & sobel operator define'''
# @torch.jit.script
class SobelConv2d(torch.jit.ScriptModule):
# class SobelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % 4 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
            self.bias_g = nn.Parameter(torch.zeros(size=(1,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias_g = None
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        # Initialize the Sobel kernal
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2
        
        # Define the trainable sobel factor

        '''gaussian filtering'''
        self.gksize = 3

        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
            self.sigma_grid = nn.Parameter(torch.ones(size=(1, 1, self.gksize,self.gksize), dtype=torch.float32), 
                                            requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)
            self.sigma_grid = nn.Parameter(torch.ones(size=(1, 1, self.gksize,self.gksize), dtype=torch.float32), 
                                            requires_grad=False)

        self.gkernel = nn.Parameter(torch.reshape(torch.FloatTensor([[0.1353, 0.3679, 0.1353],
                                                                [0.3679, 1.0000, 0.3679],
                                                                [0.1353, 0.3679, 0.1353]]), 
                                                                ((1,)+(1,) + (self.gksize,self.gksize))), requires_grad=False)

    @torch.jit.export
    def forward(self, x):

        '''gassian filtering'''
        ######################### GPU or CPU ############################# 
        # if torch.cuda.is_available():
        # self.sobel_factor = self.sobel_factor.cuda()
        self.sobel_factor = self.sobel_factor
        # self.sigma_grid = self.sigma_grid.cuda()
        self.sigma_grid = self.sigma_grid
        # if isinstance(self.bias, nn.Parameter):
        #     self.bias = self.bias.cuda()
        #########################################################
        gkernel = self.gkernel / self.sigma_grid
        gkernel /= gkernel.sum()
        sobel_weight = self.sobel_weight * self.sobel_factor
        ####################### GPU or CPU ##################################
        # if torch.cuda.is_available():
        # gkernel = gkernel.cuda()
        # sobel_weight = sobel_weight.cuda()
        #########################################################
        x = F.conv2d(x, gkernel, self.bias_g, stride=1, padding = 1, dilation=1, groups = 1) # gaussian
        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups) # sobel
        return out
#########################################################################################################
#%%
'''model define'''
# @torch.jit.script
# for torch jit, avoid using other pakages except torch (especially numpy)
class EUDnCNN(torch.jit.ScriptModule):
    def __init__(self, D, C = 32):
        super(EUDnCNN, self).__init__()
        self.D = D
        self.C = C
        # convolution layers
        self.conv = nn.ModuleList()
        self.conv.extend([nn.Conv2d(2*C+1, C, 3, padding=1) for _ in range(D)])
        self.conv.append(nn.Conv2d(2*C+1, 1, 3, padding=1))

        # apply He's initialization
        # for i in range(1, len(self.conv[:-1])):
        #     nn.init.kaiming_normal_(self.conv[i].weight.data, nonlinearity='relu')
        
        # batch normalization
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(C, float(C)).cuda() for _ in range(D)])

        #########################################################
        #########################################################

        # initialize the weights of the Batch normalization layers
        for i in range(D):
            nn.init.constant_(self.bn[i].weight.data, 1.25 * math.sqrt(C))
            # self.bn[i] = self.bn[i].cuda()
        # the first sobel convolution is added instead of first conv layer
        self.conv_sobel1 = SobelConv2d(1, 2*C, 3, padding=1)
        self.conv_sobel = SobelConv2d(1, C, kernel_size=3, stride=1, padding=1, bias=True)
    
    # @torch.jit.script
    @torch.jit.export
    def forward(self, x): 
        
        D = self.D
        h = self.conv_sobel1(x)
        h = torch.cat((x, h), dim = -3)

        '''get shapes'''
        # jit annotations (Dict) for appending
        h_buff = torch.jit.annotate(Dict[int, torch.Tensor], {}) 
        idx_buff = torch.jit.annotate(Dict[int, torch.Tensor], {})
        shape_buff_0 = torch.jit.annotate(Dict[int, int], {})
        shape_buff_1 = torch.jit.annotate(Dict[int, int], {})
        shape_buff_2 = torch.jit.annotate(Dict[int, int], {})
        shape_buff_3 = torch.jit.annotate(Dict[int, int], {})
        sub = D//2-1
        
        for i, (cbn, cconv) in enumerate(zip(self.bn, self.conv)): # indexing module list is not allowed in torch script
            #  So, using enumerate with zip
            
            if i < D//2-1:
                # save shape_buffs in jit annotatiton (Dict)
                shape_buff_0[i] = h.shape[0]; shape_buff_1[i] = h.shape[1]; 
                shape_buff_2[i] = h.shape[2]; shape_buff_3[i] = h.shape[3]

                shape_buff_0[i-sub] = h.shape[0]; shape_buff_1[i-sub] = h.shape[1]
                shape_buff_2[i-sub] = h.shape[2]; shape_buff_3[i-sub] = h.shape[3]

                h, idx = F.max_pool2d(F.relu(cbn(cconv(h))), 
                                    kernel_size=(2,2), return_indices=True)
                input_im = x.clone() 
                input_im.resize_((h.shape[0], 1, h.shape[2], h.shape[3]))
                sobeled = self.conv_sobel(input_im)
                sobel_op = torch.cat((input_im, sobeled), dim = -3)
                h = torch.cat((sobel_op, h), dim = -3)

                h_buff[i] = h; h_buff[i-sub] = h
                idx_buff[i] = idx; idx_buff[i-sub] = idx

            elif D//2-1 <= i < D//2+1:
                h = F.relu(cbn(cconv(h)))
                input_im = x.clone()
                input_im.resize_((h.shape[0],1, h.shape[2], h.shape[3]))
                sobeled = self.conv_sobel(input_im)
                sobel_op = torch.cat((input_im, sobeled), dim = -3)
            
                h = torch.cat((sobel_op, h), dim = -3)

            else:
                j = i - (D//2 + 1) + 1
                out_sz = (shape_buff_0[-j], shape_buff_1[-j],shape_buff_2[-j],shape_buff_3[-j])
                h = F.max_unpool2d(F.relu(cbn(cconv((h + h_buff[-j])/math.sqrt(2)))), 
                                idx_buff[-j], kernel_size=(2,2), output_size = out_sz )
                
                input_im = x.clone()
                input_im.resize_((h.shape[0],1, h.shape[2], h.shape[3]))
                sobeled = self.conv_sobel(input_im)
                sobel_op = torch.cat((input_im, sobeled), dim = -3)
                
                h = torch.cat((sobel_op, h), dim = -3)  

        for i, cconv in enumerate(self.conv):
            if i == D:
                y = cconv(h) + x
                return y

#%%
'''Model'''
dilate_convs = 8
model = EUDnCNN(dilate_convs).to(device)
#%%
'''pretrained model load'''
# root = 'C:\\Users\\Osstem\\Desktop\\LHW\\python38\\DNCNN\\Checkpoint\\EUDnCNN\\chest_presets\\chest3_poisson_and_gauss_compoundloss_0.1resnet18weight_8depth_scheduler50_sobel_or_gauss'
root = ''
model_dir = os.path.join(root, 'models')
model_path = os.path.join(model_dir, '13th_epoch_21.33psnr_0.9483ssim.pth') # selected
model_saved = torch.load(model_path)

#%% 
'''parameter key and value matching between jit model and pytorch based pretrianed model'''

msd = copy.deepcopy(model_saved['model_state_dict'])
D = dilate_convs
convsobel1 = ['conv_sobel1.bias', 'conv_sobel1.bias_g', 'conv_sobel1.sobel_weight', 
'conv_sobel1.sobel_factor', 'conv_sobel1.sigma_grid', 'conv_sobel1.gkernel']
forcopyconv0 = ['conv.0.bias', 'conv.0.bias_g', 'conv.0.sobel_weight', 
'conv.0.sobel_factor', 'conv.0.sigma_grid', 'conv.0.gkernel']

for i in range(len(convsobel1)):
    msd[convsobel1[i]] = copy.deepcopy(msd[forcopyconv0[i]])
    del msd[forcopyconv0[i]]

for i in range(1,10):
    value1 = copy.deepcopy(msd['conv.' + str(i) + '.weight'])
    value2 = copy.deepcopy(msd['conv.' + str(i) + '.bias'])

    del msd['conv.' + str(i) + '.weight']
    del msd['conv.' + str(i) + '.bias']

    msd['conv.' + str(i-1) + '.weight'] = value1
    msd['conv.' + str(i-1) + '.bias'] = value2

'''pretrained parameters 모델에 불러오기'''

model.load_state_dict(msd)
#%%
# '''change batch norm paramter to float type'''
# for i in range(len(model.bn)):
#     model.bn[i].eps = float(model.bn[i].eps)
#%%
#####################################################################
# model = nn.Sequential(eudncnn)
######################################################################

#%%
'''example input raw file (uint16) and normalization'''
input_name = 'preset95_new_ji'
# input_name = 'preset62'
img_path = "C:\\Users\\Osstem\\Desktop\\LHW\\ceph_imagcode\\v.1.0.2.7_Preset Tool v1.1.0\\all_presets\\" + input_name + ".raw"
out_path = "C:\\Users\\Osstem\\Desktop\\LHW\\python38\\DNCNN\\jit_model_outputs\\" + input_name + "_DN.raw"

data_dir = os.path.join(img_path)
fid=open(data_dir,"rb")
Img=np.fromfile(fid,dtype = 'uint16', sep="") # uint16 (Unsigned short in C++)
Img=np.reshape(Img, [1, 2400,3000])   # raw stitched data size (ceph)

Img_f = (Img/65535.0).astype('float')

#%%
'''convert to Tensor and inference'''
x = torch.Tensor(Img_f) # to tensor
# normalize same as training process

norm = tv.transforms.Compose([
    tv.transforms.Normalize((.5,),(.5,))]) 
x = norm(x)
model.eval()

############################################
for i in range(len(model.bn)):
    model.bn[i].eval()
#############################################

img_ = x.unsqueeze(0)
img_ = img_.to(device)

# for cliping 
def normalize16(I): 
    I *= 65535
    I[I < 0] = 0.0; I[I>65535] = 65535.0
    return I.astype(np.uint16)


with torch.no_grad():
    output = model.forward(img_)
    output2 = model.forward(output)

output = output.cpu().squeeze().numpy()
output2 = output2.cpu().squeeze().numpy()

resultimage = 0.5*output + 0.5
resultimage2 = 0.5*output2 + 0.5
final = normalize16(resultimage)
final2 = normalize16(resultimage2)

# %%
# uint8
final8 = (final.astype('float')/65535.0)*255
final8 = final8.astype('uint8')
#%%
'''plot example image and save'''
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.namedWindow('test2', cv2.WINDOW_NORMAL)
cv2.imshow('test', final)
cv2.imshow('test2', final2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite(out_path[:-4] + 'test_0615.jpg', final8, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

# fid=open(out_path, "bw")
# final.tofile(fid)
# fid.close()
#%%
'''Convert to torch script and save as .pt'''
# script_module = torch.jit.script(model)
# model.save("./test3_0615.pt")
model.save("./test3_0615.pt")
# %%
