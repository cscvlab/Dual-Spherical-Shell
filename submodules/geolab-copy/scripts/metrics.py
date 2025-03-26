import math
import numpy as np
import torch.nn.functional as F
import torch
import lpips

def bb2d(hit):
    hit = hit.numpy()
    index = np.argwhere(hit == 1)
    return np.min(index, axis=0), np.max(index, axis=0)

def compute_psnr(x, y):
    mse = (x - y)**2
    mse_num : float = mse.mean().item()
    return -10.0 * math.log10(mse_num)

def compute_psnr_selected(x, y, hit):
    mse = ((x - y)**2)[hit == 1]
    mse_num : float = mse.mean().item()
    return -10.0 * math.log10(mse_num)

def preprocess_img(img1):
    """
    Input numpy 
    Output torch
    """
    
    img1 = torch.from_numpy(img1)
    
    img1 = img1.float()/255
    return img1

def compute_ssim(
    img0,
    img1,
    max_val=1.0,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    """This code comes from svox2 """
    """Computes SSIM from two images.
    This function was modeled after tf.image.ssim, and should produce comparable
    output.

    Args:
      img0: torch.tensor. An image of size [..., width, height, num_channels].
      img1: torch.tensor. An image of size [..., width, height, num_channels].
      max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
      filter_size: int >= 1. Window size.
      filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
      k1: float > 0. One of the SSIM dampening parameters.
      k2: float > 0. One of the SSIM dampening parameters.
      return_map: Bool. If True, will cause the per-pixel SSIM "map" to returned

    Returns:
      Each image's mean SSIM, or a tensor of individual values if `return_map`.
    """
    


    device = img0.device
    ori_shape = img0.size()
    width, height, num_channels = ori_shape[-3:]
    img0 = img0.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    img1 = img1.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    batch_size = img0.shape[0]

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size, device=device) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    # z is a tensor of size [B, H, W, C]
    filt_fn1 = lambda z: F.conv2d(
        z, filt.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1),
        padding=[hw, 0], groups=num_channels)
    filt_fn2 = lambda z: F.conv2d(
        z, filt.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1),
        padding=[0, hw], groups=num_channels)

    # Vmap the blurs to the tensor size, and then compose them.
    filt_fn = lambda z: filt_fn1(filt_fn2(z))
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0 ** 2) - mu00
    sigma11 = filt_fn(img1 ** 2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = torch.clamp(sigma00, min=0.0)
    sigma11 = torch.clamp(sigma11, min=0.0)
    sigma01 = torch.sign(sigma01) * torch.min(
        torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
    )

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = torch.mean(ssim_map.reshape([-1, num_channels*width*height]), dim=-1)
    return ssim_map if return_map else ssim

def compute_ssim_selected(
    img0,
    img1,
    hit,
    max_val=1.0,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,):
    min, max = bb2d(hit)
    return compute_ssim(img0[min[0]:max[0], min[1]:max[1]], img1[min[0]:max[0], min[1]:max[1]])


def compute_lpips(im_gt, im, device = 'cuda:0'):
    """
    The value of color is [0,255]^R
    """
    
    lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)
    lpips_ = lpips_vgg(im_gt.permute([2,0,1]).cuda().contiguous(),
                       im.permute([2,0,1]).cuda().contiguous(),
                       normalize=True).item()
    return lpips_