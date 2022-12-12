# Ceph_denoising
Denoise post processed cephalometric images
- After image enhancement for easy diagnosis, it usally comes with diverse noise pattern
- This repository is handling this problem with chest x-ray dataset due to insufficiant cephalometric images

## Dataset
- The trainset is made with chest x-ray images from Kaggle dataset
- Then the trainset was preprocessed with diverse image enhancement technique to generate the noises

## CNN architecture for denoising
- The baseline of our model is U-net shaped
- Denoising model named DNCNN was referenced for embedding residual structure to encoder
- To preserve the sharpness of input image, we employed trainable sobel operator to each featuremap
