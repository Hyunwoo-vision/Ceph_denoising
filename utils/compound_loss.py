#%%
'''packages'''
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchvision import models

import math
import cv2
import numpy as np 
import copy
#########################################################################################################
#%%
'''feature extractor define'''
class ResNet50FeatureExtractor(nn.Module):

    def __init__(self, blocks = [1, 2, 3, 4], pretrained=False, progress=True, **kwargs):
        super(ResNet50FeatureExtractor, self).__init__()
        self.model = models.resnet18(pretrained, progress, **kwargs)
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        del self.model.avgpool
        del self.model.fc
        self.blocks = blocks

    def forward(self, x):
        feats = list()

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        if 1 in self.blocks:
            feats.append(x)

        x = self.model.layer2(x)
        if 2 in self.blocks:
            feats.append(x)

        x = self.model.layer3(x)
        if 3 in self.blocks:
            feats.append(x)

        x = self.model.layer4(x)
        if 4 in self.blocks:
            feats.append(x)

        return feats
#########################################################################################################
#%%
'''Compound loss'''
class CompoundLoss(_Loss):

    def __init__(self, blocks=[1, 2, 3, 4], mse_weight=1, resnet_weight=0.01):
        super(CompoundLoss, self).__init__()

        self.mse_weight = mse_weight
        self.resnet_weight = resnet_weight

        self.blocks = blocks
        self.model = ResNet50FeatureExtractor(pretrained=True)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.criterion = nn.MSELoss()

    def forward(self, input, target):
        loss_value = 0
        input = torch.cat((input, input, input), 1)
        target = torch.cat((target, target, target), 1)
        # print(input.shape)
        input_feats = self.model(input)
        target_feats = self.model(target)
        
        feats_num = len(self.blocks)
        for idx in range(feats_num):
            loss_value += self.criterion(input_feats[idx], target_feats[idx])
        loss_value /= feats_num

        loss = self.mse_weight * self.criterion(input, target) + self.resnet_weight * loss_value

        return loss
#########################################################################################################
#%%
'''SSIM define'''
def SSIM(img1, img2):
    # 0~1 mapping
    img1 = copy.deepcopy(img1.detach().cpu().numpy())
    img2 = copy.deepcopy(img2.detach().cpu().numpy())
    ssim_means = 0

    if img1.min() < 0: img1 -= img1.min()
    if img2.min() < 0: img2 -= img2.min()

    img1 /= img1.max(); img2 /= img2.max()
    img1 = img1.astype('float64'); img2 = img2.astype('float64')
    ############## 
    C1 = 0.01*2; C2 = 0.01*2
    size = 11; sigma = 1.5
    kernel = cv2.getGaussianKernel(size, sigma)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # ssim_means += ssim_map.mean()
    return ssim_map.mean()
    # return ssim_means/3