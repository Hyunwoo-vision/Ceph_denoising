#%%
import os
import cv2
import numpy as np
from skimage.util import random_noise 
import torchvision as tv
# %%
'''Dataset Class'''
class NoisyDataset(td.Dataset):
    def __init__(self, root_dir, mode = 'train', image_size=(500, 500), sigma=30):
        super(NoisyDataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.sigma = sigma
        self.images_dir = os.path.join(root_dir, mode)
        self.files = sorted(os.listdir(os.path.join(self.images_dir, 'input')))
        self.labelfiles = sorted(os.listdir(os.path.join(self.images_dir, 'label')))

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return "NoisyDataset(mode={}, image_size={}, sigma={})". \
            format(self.mode, self.image_size, self.sigma)

    def __getitem__(self, idx): 
        img_path = os.path.join(self.images_dir, 'input', self.files[idx])
        clean_path = os.path.join(self.images_dir, 'label', self.labelfiles[idx])
        clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE).astype('float64')
        clean /= clean.max()

        flag = np.random.choice(2,1); flag2 = np.random.choice(2,1)
        nimage = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype('float64')
        nimage /= nimage.max()
        poissoned = random_noise(nimage, mode = 'poisson')   
        poisson = poissoned - nimage 

        noisy = poisson*1 + nimage
        noisy /= noisy.max()
        if flag == 0:
            noisy = cv2.GaussianBlur(noisy, ksize=(3,3), sigmaX=3)
        if flag2 == 0:
            noisy = random_noise(noisy, mode = 'gaussian')

        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((.5), (.5)),
            ])

        clean = transform(clean)
        noisy = transform(noisy)

        return noisy, clean  