#%%
import torch.nn.functional as F
from torch import nn
import torch
#%%
'''sobel operator'''
class SobelConv2d(nn.Module):

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
        self.s = (self.gksize - 1) / 2
        self.t = (self.gksize - 1) / 2
        self.y, self.xx = np.ogrid[-self.s:self.s + 1, -self.t:self.t + 1]
        self.y, self.xx = torch.Tensor(self.y), torch.Tensor(self.xx)

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

        self.gkernel = nn.Parameter(torch.reshape(torch.exp(-(self.xx * self.xx + self.y * self.y)), ((1,)+(1,) + (self.gksize,self.gksize))), requires_grad=False)

    def forward(self, x):
        # print(self.sigma_grid)
        '''gassian filtering'''
        # for i in range(gaussian_kernel.shape[0]):
        #     gaussian_kernel[i, 0, :, :] /= gaussian_kernel[i, 0, :, :].sum()
        
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.cuda()
            self.sigma_grid = self.sigma_grid.cuda()
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.cuda()

        gkernel = self.gkernel / self.sigma_grid
        gkernel /= gkernel.sum()
        sobel_weight = self.sobel_weight * self.sobel_factor

        if torch.cuda.is_available():
            gkernel = gkernel.cuda()
            sobel_weight = sobel_weight.cuda()

        x = F.conv2d(x, gkernel, self.bias_g, stride=1, padding = 1, dilation=1, groups = 1)
        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out