# Ceph_denoising
Denoise post processed cephalometric images
- After image enhancement for easy diagnosis, it usally comes with diverse noise pattern
- This repository is handling this problem with chest x-ray dataset due to insufficiant cephalometric images



## Dataset
- The trainset is made with chest x-ray images from Kaggle dataset
- Then the trainset was preprocessed with diverse image enhancement technique to generate the noises
- To make corresponding labels, noisy images were denoised with pretrained Real-ESRGAN model which is SOTA for super-resolution with high denoising performance
- The reason why we apply the Real-ESRGAN model directly to our test image is that it was fit with small sized dataset
- So when we applied it our test image, it enhanced the noises regarding it as important structure
- But when we applied it to chest x-ray images, the denoising performance was really good
- The example of dataset (each enhancement result) were displayed in below

![example1](https://user-images.githubusercontent.com/65393045/206981235-1114622d-c9c1-4259-b31b-88a40c1c15ac.png)
![example21](https://user-images.githubusercontent.com/65393045/206981246-dea2e817-709e-40cf-96cf-519bcd84074b.png)
![example4](https://user-images.githubusercontent.com/65393045/206997324-ed535bdc-90cc-4e2a-99f8-addea7638dcf.png)



## CNN architecture for denoising
- The baseline of our model is U-net shaped
- Denoising model named DNCNN was referenced for embedding residual structure to encoder
- To preserve the sharpness of input image, we employed trainable sobel operator to each featuremap
- The example of model architecture is displayed in below

![example3](https://user-images.githubusercontent.com/65393045/206994032-617fc8b6-4ea9-45d3-9bbc-78fddb263691.png)



## Training detail

![example9](https://user-images.githubusercontent.com/65393045/207000103-50fdc481-8a37-4843-bc68-87a830855cd5.png)



## Results

![example8](https://user-images.githubusercontent.com/65393045/206998985-b14232b1-a4ec-48b9-80d8-f3ac194dddbe.png)


