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

def save_flows(flows,root,name,itype="npz"):
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

def _save_flows(flows,root,name,itype="npz"):

    # -- path --
    root = Path(str(root))
    if not root.exists():
        print(f"Making dir for save_vid [{str(root)}]")
        root.mkdir(parents=True)
    assert root.exists()

    # -- path --
    path = root / ("%s.npz" % name)

    # -- unpack --
    fflow = flows['fflow']
    bflow = flows['bflow']

    # -- save --
    np.savez_compressed(path,fflow=fflow,bflow=bflow)

def read_flows(root,name):

    # -- path --
    path = root / ("%s.npz" % name)

    # -- read --
    flows_npz = np.load(path)#,fflows=fflows,bflows=bflows)

    # -- format --
    flows = edict({"fflow":flows_npz['fflow'],
                   "bflow":flows_npz['bflow']})

    return flows


