import numpy as np
import torch as th
from PIL import Image
from pathlib import Path
import rawpy,imageio
from einops import rearrange
from skimage import exposure
from easydict import EasyDict as edict


def lin2rgb(im):
    """ Convert im from "Linear sRGB" to sRGB - apply Gamma. """
    # sRGB standard applies gamma = 2.4,
    # and Break Point = 0.00304 (and computed Slope = 12.92)
    g = 2.2
    bp = 0.00304
    inv_g = 1/g
    sls = 1 / (g/(bp**(inv_g - 1)) - g*bp + bp)
    fs = g*sls / (bp**(inv_g - 1))
    co = fs*bp**(inv_g) - sls*bp

    srgb = im.copy()
    srgb[im <= bp] = sls * im[im <= bp]
    srgb[im > bp] = np.power(fs*im[im > bp], inv_g) - co
    return srgb

def select_mat_key(info):
    opts = ["noisy","noisy_list"]
    for opt in opts:
        if opt in info: return opt
    raise KeyError("Uknown key for .mat file.")

def read_mat(filename):
    import scipy
    info = scipy.io.loadmat(filename)
    key = select_mat_key(info)
    noisy = info[key].astype('float32')#/2**16
    # save_raw(noisy)
    return noisy

def save_raw(noisy):
    import matplotlib.pyplot as plt
    plt.figure(figsize = (20,10))
    plt.imshow(noisy[0,...,:3])
    plt.axis('off');
    plt.title('raw noisy clip');
    plt.savefig("this.png")
    # exit()

def read_raw(filename,clip=True):

    # -- shortcut .mat file for now --
    if "mat" in Path(filename).suffix:
        return read_mat(filename),{}

    # -- read raw --
    _raw = rawpy.imread(filename)
    blacklevel = np.mean(_raw.black_level_per_channel)
    whitelevel = _raw.white_level
    xyz2cam = _raw.rgb_xyz_matrix[:3,:3]
    raw = _raw.raw_image_visible

    # -- output linearized --
    print(blacklevel)
    raw = (raw.astype(np.float64) - blacklevel) / (whitelevel - blacklevel)
    if clip:
        raw = raw.clip(0, 1)

    # -- save info --
    info = {"xyz2cam":xyz2cam}
    return raw,edict(info)

def file_raw2rgb(filename):
    raw,info = read_raw(filename)
    rgb = raw2rgb(raw,info.xyz2cam)
    return rgb

def video_raw2rgb(vid,info=None,contrast_fxn=None):
    B,T,F,H,W = vid.shape
    vid_rgb = []
    for bi in range(vid.shape[0]):
        for ti in range(vid.shape[1]):
            raw_bt = vid[bi,ti]
            info_bt = info[bi][ti] if not(info is None) else None
            # assert raw_bt.shape[0] == 1,"No Channels."
            raw_bt = raw_bt[0] # no channels
            rgb_bt = raw2rgb(raw_bt,info_bt,contrast_fxn=contrast_fxn)
            rgb_bt = rearrange(rgb_bt,'h w c -> c h w')
            vid_rgb.append(rgb_bt)
    H,W = vid_rgb[-1].shape[-2:]
    vid_rgb = np.stack(vid_rgb).reshape(B,T,3,H,W)
    return vid_rgb

def raw2rgb(raw,xyz2cam=None,rescale=True,contrast_fxn=None):
    # _raw = rawpy.imread(filename)
    # print(dir(_raw))
    if th.is_tensor(raw):
        raw = raw.detach().cpu().numpy()

    # filename = 'sample__lin_bayer.tif'  # Output of: dcraw -4 -D -T sample.dng

    # -- colormatrix --
    if xyz2cam is None:
        xyz2cam = np.array([[ 0.6722, -0.0635, -0.0963],
                            [-0.4287,  1.2460,  0.2028],
                            [-0.0908,  0.2162,  0.5668]])
    # Constant matrix for converting sRGB to XYZ(D65):
    # http://www.brucelindbloom.com/Eqn_RGB_XYZ_Matrix.html
    rgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])

    # Exif information:
    if rescale: AsShotNeutral = np.array([0.5185, 1, 0.5458])
    else: AsShotNeutral = np.array([1, 1, 1])
    wb_multipliers = 1 / AsShotNeutral
    r_scale = wb_multipliers[0]  # Assume value is above 1
    g_scale = wb_multipliers[1]  # Assume value = 1
    b_scale = wb_multipliers[2]  # Assume value is above 1

    # Bayer alignment is RGGB:
    # R G
    # G B
    #
    # Apply white balancing to linear Bayer image.
    balanced_bayer = raw.copy()
    balanced_bayer[0::2, 0::2] = balanced_bayer[0::2, 0::2]*r_scale  # Red   (indices [0, 2, 4,... ; 0, 2, 4,... ])
    balanced_bayer[0::2, 1::2] = balanced_bayer[0::2, 1::2]*g_scale  # Green (indices [0, 2, 4,... ; 1, 3, 5,... ])
    balanced_bayer[1::2, 0::2] = balanced_bayer[1::2, 0::2]*g_scale  # Green (indices [1, 3, 5,... ; 0, 2, 4,... ])
    balanced_bayer[1::2, 1::2] = balanced_bayer[1::2, 1::2]*b_scale  # Blue  (indices [1, 3, 5,... ; 0, 2, 4,... ])

    # Clip to range [0, 1] for avoiding "pinkish highlights" (avoiding "magenta cast" in the highlights).
    balanced_bayer = np.minimum(balanced_bayer, 1)

    # Demosaicing:
    temp = np.round((balanced_bayer*(2**16-1))).astype(np.uint16)  # Convert from double to np.uint16, because OpenCV demosaic() function requires a uint8 or uint16 input.
    import cv2
    lin_rgb = cv2.cvtColor(temp, cv2.COLOR_BayerBG2RGB).astype(np.float64)/(2**16-1)  # Apply Demosaicing and convert back to np.float64 in range [0, 1] (is there a bug in OpenCV Bayer naming?).

    # Color Space Conversion
    # xyz2cam = raw_info.xyz2cam  # ColorMatrix applies XYZ(D65) to CAM_rgb
    rgb2cam = xyz2cam @ rgb2xyz

    # Result:
    # rgb2cam = [0.2619    0.1835    0.0252
    #            0.0921    0.7620    0.2053
    #            0.0195    0.1897    0.5379]

    # Normalize rows to 1. MATLAB shortcut: rgb2cam = rgb2cam ./ repmat(sum(rgb2cam,2),1,3);

    rows_sum = np.sum(rgb2cam, 1)
    # Result:
    # rows_sum = [0.4706
    #             1.0593
    #             0.7470]

    # Divide element of every row by the sum of the row:
    rgb2cam[0, :] = rgb2cam[0, :] / rows_sum[0]  # Divide top row
    rgb2cam[1, :] = rgb2cam[1, :] / rows_sum[1]  # Divide center row
    rgb2cam[2, :] = rgb2cam[2, :] / rows_sum[2]  # Divide bottom row
    # Result (sum of every row is 1):
    # rgb2cam = [0.5566    0.3899    0.0535
    #            0.0869    0.7193    0.1938
    #            0.0261    0.2539    0.7200]

    cam2rgb = np.linalg.inv(rgb2cam)  # Invert matrix
    # Result:
    # cam2rgb = [ 1.9644   -1.1197    0.1553
    #            -0.2412    1.6738   -0.4326
    #             0.0139   -0.5498    1.5359]

    r = lin_rgb[:, :, 0]
    g = lin_rgb[:, :, 1]
    b = lin_rgb[:, :, 2]

    # Left multiply matrix cam2rgb by each RGB tuple (convert from "camera RGB" to "linear sRGB").
    sr = cam2rgb[0, 0]*r + cam2rgb[0, 1]*g + cam2rgb[0, 2]*b
    sg = cam2rgb[1, 0]*r + cam2rgb[1, 1]*g + cam2rgb[1, 2]*b
    sb = cam2rgb[2, 0]*r + cam2rgb[2, 1]*g + cam2rgb[2, 2]*b

    lin_srgb = np.dstack([sr, sg, sb])
    lin_srgb = lin_srgb.clip(0, 1)  # Clip to range [0, 1]

    # Convert from "Linear sRGB" to sRGB (apply gamma)
    sRGB = lin2rgb(lin_srgb)  # lin2rgb MATLAB functions uses the exact formula [you may approximate it to power of (1/gamma)].

    # Contrast stretching
    if contrast_fxn is None:
        p2, p98 = np.percentile(sRGB, (2, 98))
        sRGB = exposure.rescale_intensity(sRGB, in_range=(p2, p98))
    else:
        sRGB = contrast_fxn(sRGB)

    return sRGB

# Save to sRGB.png
# filename = "data/curtains_v2/IMG_1306.CR2"
# filename = "data/shelf_bottles/IMG_1350.CR2"
# filename = "data/floor/IMG_1294.CR2"
def main():
    filename = "data/calibrate/IMG_1329.CR2"
    sRGB = file_raw2rgb(filename)
    import cv2
    cv2.imwrite('sRGB.png', cv2.cvtColor((sRGB*255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def vid_packing(vid,mode):
    B,T,_,H,W = vid.shape
    xform = []
    for bi in range(vid.shape[0]):
        for ti in range(vid.shape[1]):
            vid_bt = vid[bi,ti]
            if th.is_tensor(vid_bt):
                vid_bt = vid_bt.detach().cpu().numpy()
            xform.append(packing(vid_bt,mode))
    H,W = xform[-1].shape[-2:]
    xform = np.stack(xform).reshape(B,T,-1,H,W)
    if th.is_tensor(vid):
        xform = th.from_numpy(xform).to(vid.device)
    return xform

def packing(rgb_or_raw,mode):
    if mode == 'raw2rgb':
        return packing_raw2rgb(rgb_or_raw)
    elif mode == 'rgb2raw':
        return packing_rgb2raw(rgb_or_raw)
    else:
        raise ValueError(f"Uknown packing mode [{mode}]")

def packing_raw2rgb(raw):

    # -- scaling values --
    AsShotNeutral = np.array([0.5185, 1, 0.5458])
    wb_multipliers = 1 / AsShotNeutral
    r_scale = wb_multipliers[0]  # Assume value is above 1
    g_scale = wb_multipliers[1]  # Assume value = 1
    b_scale = wb_multipliers[2]  # Assume value is above 1

    # -- unpack to scaled --
    raw = raw.copy()
    red = raw[0::2, 0::2]*r_scale
    green0 = raw[0::2, 1::2]*g_scale
    green1 = raw[1::2, 0::2]*g_scale
    blue = raw[1::2, 1::2]*b_scale

    # -- stack and clip --
    img = np.stack([red,green0,blue,green1],0)
    img = np.minimum(img, 1)

    return img

def packing_rgb2raw(rgb):

    # -- allocate --
    pH,pW = rgb.shape[-2:]
    H,W = 2*pH,2*pW
    unpacked = np.zeros((H,W),dtype=np.float64)

    # -- scaling values --
    AsShotNeutral = np.array([0.5185, 1, 0.5458])
    wb_multipliers = 1 / AsShotNeutral
    r_scale = wb_multipliers[0]  # Assume value is above 1
    g_scale = wb_multipliers[1]  # Assume value = 1
    b_scale = wb_multipliers[2]  # Assume value is above 1

    # -- unpack to scaled --
    if rgb.shape[0] == 3:
        red,green0,blue,green1 = rgb[0],rgb[1],rgb[2],rgb[1]
    else:
        red,green0,blue,green1 = rgb[0],rgb[1],rgb[2],rgb[3]
    unpacked[0::2, 0::2] = red/r_scale
    unpacked[0::2, 1::2] = green0/g_scale
    unpacked[1::2, 0::2] = green1/g_scale
    unpacked[1::2, 1::2] = blue/b_scale

    return unpacked

if __name__ == "__main__":
    main()


