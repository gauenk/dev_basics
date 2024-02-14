import torch as th
import numpy as np
from einops import rearrange
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio as comp_psnr
from skimage.metrics import structural_similarity as compute_ssim_ski
from skvideo.measure import strred as comp_strred

def compute_batched(compute_fxn,clean,deno,div=255.):
    metric = []
    for b in range(len(clean)):
        metric_b = compute_fxn(clean[b],deno[b],div)
        metric.append(metric_b)
    metric = np.array(metric)
    return metric

def compute_ssims(clean,deno,div=1.):
    # -- optional batching --
    if clean.ndim == 5:
        return compute_batched(compute_ssims,clean,deno,div)

    # -- to numpy --
    clean = clean.detach().cpu().numpy()
    deno = deno.detach().cpu().numpy()

    # -- standardize image --
    # if np.isclose(div,1.):
    #     deno = deno.clip(0,1)*255.
    #     clean = clean.clip(0,1)*255.
    #     deno = deno.astype(np.uint8)/255.
    #     clean = clean.astype(np.uint8)/255.
    # elif np.isclose(div,255.):
    #     deno = deno.astype(np.uint8)*1.
    #     clean = clean.astype(np.uint8)*1.

    nframes = clean.shape[0]
    ssims = []
    for t in range(nframes):
        clean_t = clean[t].transpose((1,2,0))/div
        deno_t = deno[t].transpose((1,2,0))/div
        ssim_t = compute_ssim_ski(clean_t,deno_t,channel_axis=-1,
                                  data_range=1.)
        ssims.append(ssim_t)
    ssims = np.array(ssims)
    return ssims

def compute_psnrs(clean,deno,div=1.):
    # -- optional batching --
    if clean.ndim == 5:
        return compute_batched(compute_psnrs,clean,deno,div)
    t = clean.shape[0]

    # -- to numpy --
    clean = clean.detach().cpu().numpy()
    deno = deno.detach().cpu().numpy()

    # -- standardize image --
    # if np.isclose(div,1.):
    #     deno = deno.clip(0,1)*255.
    #     clean = clean.clip(0,1)*255.
    #     deno = deno.astype(np.uint8)/255.
    #     clean = clean.astype(np.uint8)/255.
    # elif np.isclose(div,255.):
    #     deno = deno.astype(np.uint8)*1.
    #     clean = clean.astype(np.uint8)*1.

    psnrs = []
    t = clean.shape[0]
    for ti in range(t):
        psnr_ti = comp_psnr(clean[ti,:,:,:], deno[ti,:,:,:], data_range=div)
        psnrs.append(psnr_ti)
    return np.array(psnrs)

def compute_strred(clean,deno,div=1):

    # -- optional batching --
    if clean.ndim == 5:
        return compute_batched(compute_strred,clean,deno,div)

    # -- numpify --
    clean = clean.detach().cpu().numpy()
    deno = deno.detach().cpu().numpy()

    # -- reshape --
    clean = rearrange(clean,'t c h w -> t h w c')/float(div)
    deno = rearrange(deno,'t c h w -> t h w c')/float(div)

    # -- bw --
    if clean.shape[-1] == 3:
        clean = rgb2gray(clean,channel_axis=-1)
        deno = rgb2gray(deno,channel_axis=-1)

    # -- compute --
    with np.errstate(invalid='ignore'):
        outs = comp_strred(clean,deno)
    strred = outs[1] # get float
    return strred

def _blocking_effect_factor(im):
    im = th.from_numpy(im)

    block_size = 8

    block_horizontal_positions = th.arange(7, im.shape[3] - 1, 8)
    block_vertical_positions = th.arange(7, im.shape[2] - 1, 8)

    horizontal_block_difference = (
                (im[:, :, :, block_horizontal_positions] - im[:, :, :, block_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_block_difference = (
                (im[:, :, block_vertical_positions, :] - im[:, :, block_vertical_positions + 1, :]) ** 2).sum(3).sum(
        2).sum(1)

    nonblock_horizontal_positions = np.setdiff1d(th.arange(0, im.shape[3] - 1), block_horizontal_positions)
    nonblock_vertical_positions = np.setdiff1d(th.arange(0, im.shape[2] - 1), block_vertical_positions)

    horizontal_nonblock_difference = (
                (im[:, :, :, nonblock_horizontal_positions] - im[:, :, :, nonblock_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_nonblock_difference = (
                (im[:, :, nonblock_vertical_positions, :] - im[:, :, nonblock_vertical_positions + 1, :]) ** 2).sum(
        3).sum(2).sum(1)

    n_boundary_horiz = im.shape[2] * (im.shape[3] // block_size - 1)
    n_boundary_vert = im.shape[3] * (im.shape[2] // block_size - 1)
    boundary_difference = (horizontal_block_difference + vertical_block_difference) / (
                n_boundary_horiz + n_boundary_vert)

    n_nonboundary_horiz = im.shape[2] * (im.shape[3] - 1) - n_boundary_horiz
    n_nonboundary_vert = im.shape[3] * (im.shape[2] - 1) - n_boundary_vert
    nonboundary_difference = (horizontal_nonblock_difference + vertical_nonblock_difference) / (
                n_nonboundary_horiz + n_nonboundary_vert)

    scaler = np.log2(block_size) / np.log2(min([im.shape[2], im.shape[3]]))
    bef = scaler * (boundary_difference - nonboundary_difference)

    bef[boundary_difference <= nonboundary_difference] = 0
    return bef
