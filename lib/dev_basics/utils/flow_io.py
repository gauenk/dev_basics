# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- file io --
from PIL import Image
from pathlib import Path
from easydict import EasyDict as edict

# -- local --
from .vid_io import mangle_fn

def save_flows(flows,root,name,itype="npy"):
    ndim = flows.fflow.ndim
    nbatch = flows.fflow.shape[0]
    if ndim == 4:
        return _save_flows(flows,root,name,itype)
    elif ndim == 5 and nbatch == 1:
        flows = {k:v[0] for k,v in flows.items()}
        return _save_flows(flows,root,name,itype)
    elif ndim == 5 and nbatch > 1:
        fns = []
        for b in range(nbatch):
            flows_b = {k:v[b] for k,v in flows.items()}
            fns_b = _save_flows(flows_b,root,"%s_%02d" % (name,b),itype)
            fns.extend(fns_b)
        return fns
    else:
        raise ValueError("Uknown tensor sizing [%d] and [%d]" % (ndim,nbatch))

def _save_flows(flows,root,name,itype="npy"):

    # -- path --
    root = Path(str(root))
    if not root.exists():
        print(f"Making dir for save_vid [{str(root)}]")
        root.mkdir(parents=True)
    assert root.exists()

    # -- save --
    if itype == "npz":
        _save_flows_npz(flows,root,name)
    elif itype == "npy":
        _save_flows_npy(flows,root,name)

def _save_flows_npy(flows,root,name):

    # -- save --
    path = root / ("%s_fflow.npy" % name)
    np.save(path,flows['fflow'])
    path = root / ("%s_bflow.npy" % name)
    np.save(path,flows['bflow'])

def _save_flows_npz(flows,root,name):

    # -- path --
    path = root / ("%s.npz" % name)

    # -- unpack --
    fflow = flows['fflow']
    bflow = flows['bflow']

    # -- save --
    np.savez_compressed(path,fflow=fflow,bflow=bflow)

def read_flows(root,name,mmap_mode=None,itype="npy"):

    if itype == "npz":
        return _read_flows_npz(root,name,mmap_mode)
    elif itype == "npy":
        return _read_flows_npy(root,name,mmap_mode)
    else:
        raise ValueError(f"Uknown flow io type [{itype}]")

def _read_flows_npz(root,name,mmap_mode=None):

    # -- path --
    root = Path(root)
    path = root / ("%s.npz" % name)
    flow = np.load(path,mmap_mode=mmap_mode)

    # -- format --
    flows = edict({"fflow":flow['fflow'],
                   "bflow":flow['bflow']})

    return flows


def _read_flows_npy(root,name,mmap_mode=None):

    # -- path --
    root = Path(root)
    path = root / ("%s_fflow.npy" % name)
    fflow = np.load(path,mmap_mode=mmap_mode)
    path = root / ("%s_bflow.npy" % name)
    bflow = np.load(path,mmap_mode=mmap_mode)

    # -- format --
    flows = edict({"fflow":fflow,"bflow":bflow})

    return flows


