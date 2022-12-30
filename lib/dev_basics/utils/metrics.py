import torch as th
import numpy as np
from einops import rearrange
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio as comp_psnr
from skimage.metrics import structural_similarity as compute_ssim_ski
from skvideo.measure import strred as comp_strred

def compute_ssims(clean,deno,div=255.):
    nframes = clean.shape[0]
    ssims = []
    for t in range(nframes):
        clean_t = clean[t].cpu().numpy().transpose((1,2,0))/div
        deno_t = deno[t].cpu().numpy().transpose((1,2,0))/div
        ssim_t = compute_ssim_ski(clean_t,deno_t,channel_axis=-1,
                                  data_range=1.)
        ssims.append(ssim_t)
    ssims = np.array(ssims)
    return ssims

def compute_psnrs(clean,deno,div=255.):
    t = clean.shape[0]
    clean = clean.detach().cpu().numpy()
    deno = deno.detach().cpu().numpy()
    # clean_rs = clean.reshape((t,-1))/div
    # deno_rs = deno.reshape((t,-1))/div
    # mse = th.mean((clean_rs - deno_rs)**2,1)
    # psnrs = -10. * th.log10(mse).detach()
    # psnrs = psnrs.cpu().numpy()
    psnrs = []
    t = clean.shape[0]
    for ti in range(t):
        psnr_ti = comp_psnr(clean[ti,:,:,:], deno[ti,:,:,:], data_range=div)
        psnrs.append(psnr_ti)
    return np.array(psnrs)


def compute_strred(clean,deno,div=255):

    # -- numpify --
    clean = clean.detach().cpu().numpy()
    deno = deno.detach().cpu().numpy()

    # -- reshape --
    clean = rearrange(clean,'t c h w -> t h w c')/div
    deno = rearrange(deno,'t c h w -> t h w c')/div

    # -- bw --
    if clean.shape[-1] == 3:
        clean = rgb2gray(clean,channel_axis=-1)
        deno = rgb2gray(deno,channel_axis=-1)

    # -- compute --
    outs = comp_strred(clean,deno)
    strred = outs[1] # get float
    return strred

