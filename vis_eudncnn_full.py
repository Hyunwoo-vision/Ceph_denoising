#%%
'''Packages'''
import os
import numpy as np
import torch

import torchvision as tv
from PIL import Image
import cv2
import torch.autograd.profiler as profiler

from utils.EUDNCNN import *

import gc
from skimage.exposure import match_histograms
gc.collect()
torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#%%

'''M A I N script'''

#%%
'''get dataset and model'''
transform = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((.5), (.5))])
image_file = 'preset95_new_ji.jpg'
# root = 'C:\\Users\\Osstem\\Desktop\\LHW\\python38\\DNCNN\\test\\' + image_file
root = 'C:\\Users\\Osstem\\Desktop\\LHW\\python38\\DNCNN\\Checkpoint\\EUDnCNN\\chest_presets\\chest3_poisson_and_gauss_compoundloss_0.1resnet18weight_8depth_scheduler50_sobel_or_gauss\\presets_13th_epoch\\' + image_file
cephimage = cv2.imread(root, cv2.IMREAD_GRAYSCALE)
# cv2.imwrite(root[:-4] + '.jpg', cephimage, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
# cephimage = Image.open(root).convert('L')
origin = cv2.imread(root, cv2.IMREAD_GRAYSCALE)
# h,w = 700, 700
# hh, ww = cephimage.size[1]//h, cephimage.size[0]//w
# cephimage = np.array(cephimage)

#%%
'''foward propagation'''
dilate_convs = 8
model=EUDnCNN(dilate_convs).cuda()

root = 'C:\\Users\\Osstem\\Desktop\\LHW\\python38\\DNCNN\\Checkpoint\\EUDnCNN\\chest_presets\\chest3_poisson_and_gauss_compoundloss_0.1resnet18weight_8depth_scheduler50_sobel_or_gauss'
# root = 'C:\\Users\\Osstem\\Desktop\\LHW\\python38\\DNCNN\\Checkpoint\\EUDnCNN\\chest3\\chest3_poissongauss_compoundloss_0.1resnet18weight_8depth_scheduler50_sobel_or_gauss'
model_dir = os.path.join(root, 'models')
model_path = os.path.join(model_dir, '13th_epoch_21.33psnr_0.9483ssim.pth') # selected

model_saved = torch.load(model_path)

model.load_state_dict(model_saved['model_state_dict'])
model.eval()
# model.train()
#%%
###################
# cephimage = cv2.resize(cephimage, (700,700))
##################
image = Image.fromarray(cephimage)
noisy = transform(image)
noisy = torch.reshape(noisy, ((1,) + noisy.shape))

with torch.no_grad():
    with profiler.profile(with_stack=True, use_cuda=True, profile_memory=True) as prof:
        denoised = model(noisy.cuda())
print(prof.key_averages().table(sort_by= "cuda_time_total", row_limit=10))
# resultimage = denoised.squeeze(0)
resultimage = denoised.permute(0,2,3,1).detach().cpu().squeeze().numpy()
print("Allocated:", round(torch.cuda.memory_allocated(0)/10243,1), "GB")
print("Cached:", round(torch.cuda.memory_cached()/1024**3,1), "MB")
             
#%% test
# resultimage = 0.5*resultimage[0] + 0.5
resultimage = 0.5*resultimage + 0.5
resultimage = np.clip(resultimage*255, 0, 255).astype('uint8')

#############################################
# plt.imshow(resultimage.to('cpu').numpy(), cmap = 'gray')
# plt.show()
#############################################


cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.imshow('test', resultimage)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(os.path.join(root, 'presets_13th_epoch', image_file[:-4] + '_dn.jpg'), resultimage, [int(cv2.IMWRITE_JPEG_QUALITY), 100])# %%
#%%
'''plot denoised image and save'''

# if resultimage.min() < 0:
#     resultimage -= resultimage.min()
# resultimage /= resultimage.max()
# resultimage *= origin.max()
# sub = int(resultimage.mean() - origin.mean())

# if sub < -5 : 
#     sub += 5
#     resultimage -= sub
# if sub > 5: 
#     sub -= 5
#     resultimage -= sub

# '''debug test'''
# resultimage += 10
# ''''''
# resultimage = resultimage.astype('uint8')

# cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
# cv2.namedWindow('result', cv2.WINDOW_NORMAL)
# cv2.imshow('origin', origin)
# cv2.imshow('result', resultimage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imwrite(os.path.join(root, 'presets', image_file), resultimage, [int(cv2.IMWRITE_JPEG_QUALITY), 100])# %%
# %%

# %%
