# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- file io --
from PIL import Image
from pathlib import Path

def save_burst(burst,root,name):
    save_video(burst,root,name)

def save_video(vid,root,name):
    if vid.ndim == 4:
        _save_video(vid,root,name)
    elif vid.ndim == 5 and vid.shape[0] == 1:
        _save_video(vid[0],root,name)
    elif vid.ndim == 5 and vid.shape[0] > 1:
        B = vid.shape[0]
        for b in range(B):
            _save_video(vid[b],root,"%s_%02d" % (name,b))
    else:
        raise ValueError("Uknown number of dims [%d]" % vid.ndim)

def _save_video(vid,root,name):
    # -- path --
    root = Path(str(root))
    if not root.exists():
        print(f"Making dir for save_vid [{str(root)}]")
        root.mkdir(parents=True)
    assert root.exists()

    # -- save --
    save_fns = []
    nframes = vid.shape[0]
    for t in range(nframes):
        img_t = vid[t]
        path_t = root / ("%s_%05d.png" % (name,t))
        save_image(img_t,str(path_t))
        save_fns.append(str(path_t))
    return save_fns

def save_image(image,path):

    # -- to numpy --
    if th.is_tensor(image):
        image = image.detach().cpu().numpy()

    # -- rescale --
    if image.max() > 300: # probably from a fold
        image /= image.max()

    # -- to uint8 --
    if image.max() < 100:
        image = image*255.
    image = np.clip(image,0,255).astype(np.uint8)

    # -- remove single color --
    image = rearrange(image,'c h w -> h w c')
    image = image.squeeze()

    # -- save --
    img = Image.fromarray(image)
    img.save(path)

def read_files(fns):
    vid = []
    for fn in fns:
        img = np.array(Image.open(fn))
        img = rearrange(img,'h w c -> c h w')
        img = th.from_numpy(img)
        vid.append(img)
    vid = th.stack(vid)
    return vid
