#%% 
'''Packages'''
import os
from cv2 import Sobel
import numpy as np
import torch
import torch.utils.data as td

import matplotlib.pyplot as plt
import time
import cv2
import copy
import math

from torchvision.transforms.functional import to_pil_image
import torch.optim as optim
from torch.optim import lr_scheduler

from Train_and_Test import *
from utils.EUDNCNN import *
from utils.eval_metrices import *
from Dataset import *

from utils.compound_loss import *
import gc
gc.collect()
torch.cuda.empty_cache()
#%%

'''M A I N script'''


#%% 
'''get dataset / hyperparameters'''
sigma = 20
dilate_convs = 8
lr = 0.001
resnet_weight = 0.1
criterion = CompoundLoss(blocks=[1,2,3,4], mse_weight=1, resnet_weight=resnet_weight)
EPOCH = 200  
ssize = 7

root_dir = 'C:\\Users\\Osstem\\Desktop\\LHW\\python38\\DNCNN\\chest_presets\\'
model_dir = 'C:\\Users\\Osstem\\Desktop\\LHW\\python38\\DNCNN\\Checkpoint\\EUDnCNN\\chest_presets\\chest3_poisson_and_gauss_compoundloss_{}resnet18weight_{}depth_scheduler{}_sobel_or_gauss'.format(resnet_weight, dilate_convs, 50)

trainset = NoisyDataset(root_dir, mode = 'train', image_size=(700,700)) 
testset = NoisyDataset(root_dir, mode = 'test', image_size = (700,700), sigma=sigma)
#%% 
'''checkout example data'''
x = testset[5]
image_noise = 0.5*x[0]+0.5   #denormalization 해줌
image_clean = 0.5*x[1]+0.5

plt.figure(figsize=(30,15))
plt.subplot(1,2,1)
plt.imshow(to_pil_image(image_noise), cmap = 'gray')
plt.title('noisy')
plt.subplot(1,2,2)
plt.imshow(to_pil_image(image_clean), cmap = 'gray')
plt.title('clean')

cv2.namedWindow('noisy', cv2.WINDOW_NORMAL)
cv2.namedWindow('clean', cv2.WINDOW_NORMAL)
cv2.imshow('noisy', np.array(to_pil_image(image_noise)))
cv2.imshow('clean', np.array(to_pil_image(image_clean)))
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
'''get batches'''
batch_size = 8
train_loader = td.DataLoader(trainset, batch_size = batch_size, shuffle=True)
test_loader = td.DataLoader(testset, batch_size = batch_size, shuffle=False)
first_batch_train=train_loader.__iter__().__next__()
first_batch_test=test_loader.__iter__().__next__()

print('train batch shape : {}'.format(first_batch_train[0].shape))
print('test batch shape : {}'.format(first_batch_test[0].shape))
#%%
'''load checkpoint / Get model'''
if os.path.isdir(os.path.join(model_dir, "models")):
    if len(os.listdir(os.path.join(model_dir, "models"))) > 0:
        times = []
        model_list = os.listdir(os.path.join(model_dir, "models"))
        for i in model_list:
            t = os.path.getctime(os.path.join(model_dir, "models", i))
            times.append(t)
        a = sorted(range(len(times)), key=lambda k: times[k])
        model_saved = os.path.join(model_dir, "models", model_list[a[-1]])
        checkpoint = torch.load(model_saved)
        model=EUDnCNN(dilate_convs).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr = lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=ssize, gamma=0.1)
        start_epoch = checkpoint['epoch'] + 1
    else:
        model=EUDnCNN(dilate_convs)
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr = lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=ssize, gamma=0.1)
        start_epoch = 0
else:
    os.mkdir(model_dir)
    os.mkdir(os.path.join(model_dir, "models"))
    model=EUDnCNN(dilate_convs)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=ssize, gamma=0.1)
    start_epoch = 0
model
#%%
'''train'''
train_loss, test_loss = [], []
train_psnr, test_psnr = [], []
train_ssim, test_ssim = [], []
best_epoch_psnr = 0
best_epoch_ssim = 0

for epoch in range(start_epoch, EPOCH):
    start = time.time()

    print(f'Epoch {epoch + 1} of {EPOCH}')
    train_epoch_loss, train_epoch_psnr, train_epoch_ssim = train(model, train_loader, scheduler, start)
    test_epoch_loss, test_epoch_psnr, test_epoch_ssim = test(model, test_loader)

    train_loss.append(train_epoch_loss)
    train_psnr.append(train_epoch_psnr)
    train_ssim.append(test_epoch_ssim)

    test_loss.append(test_epoch_loss)
    test_psnr.append(test_epoch_psnr)
    test_ssim.append(test_epoch_ssim)

    end = time.time()

    print(f'Train PSNR: {train_epoch_psnr:.3f}, Val PSNR: {test_epoch_psnr:.3f}, Time: {(end-start)//60}m {(end-start)%60:.1f}sec')
    print(f'Train SSIM: {train_epoch_ssim:.3f}, Val SSIM: {test_epoch_ssim:.3f}, Time: {(end-start)//60}m {(end-start)%60:.1f}sec')
    print(f'Train Loss: {train_epoch_loss:.3f}, Val loss: {test_epoch_loss:.3f}, Time: {(end-start)//60}m {(end-start)%60:.1f}sec')
    # saving model
    if test_epoch_psnr >= best_epoch_psnr or test_epoch_ssim >= best_epoch_ssim:
        # best_epoch_psnr = copy.deepcopy(test_epoch_psnr)
        best_epoch_psnr = max(best_epoch_psnr, test_epoch_psnr)
        best_epoch_ssim = max(best_epoch_ssim, test_epoch_ssim)
        best_model_wts = copy.deepcopy(model)

    checkpoint_path = os.path.join(model_dir, "models", "{}th_epoch_{:.2f}psnr_{:.4f}ssim.pth".format(epoch+1, test_epoch_psnr, test_epoch_ssim))
    torch.save({
                'epoch': epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'train_loss' : train_epoch_loss,
                'test_loss' : test_epoch_loss,
                'train_psnr' : train_epoch_psnr,
                'test_psnr' : test_epoch_psnr,
                'train_ssim' : train_epoch_ssim,
                'test_ssim' : test_epoch_ssim
                },
                checkpoint_path) 

