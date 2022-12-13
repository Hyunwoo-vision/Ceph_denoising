#%%
import math
import cv2
import numpy as np
import copy
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

'''PSNR, SSIM, loss functions define'''
def PSNR(label, outputs, max_val=1.):
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()

    img_diff = outputs - label
    rmse = math.sqrt(np.mean((img_diff)**2))
    if rmse == 0: # label과 output이 완전히 일치하는 경우
        return 100
    else:
        psnr = 20 * math.log10(max_val/rmse)
        return psnr

def SSIMbatch(outputs, targets):
    b_sz = outputs.shape[0]
    ssim_sum = 0.0
    
    for i in range(b_sz):
        img1 = outputs[i,:].squeeze(); img2 = targets[i,:].squeeze()
        # print(img1.shape); print(img2.shape)
        ssim = SSIM(img1, img2)
        ssim_sum += ssim

    return ssim_sum