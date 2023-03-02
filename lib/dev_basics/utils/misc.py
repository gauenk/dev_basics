import yaml
import pickle
import random
import numpy as np
import torch as th
from easydict import EasyDict as edict

def optional(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default

def optional_delete(pydict,key):
    if pydict is None: return
    elif key in pydict: del pydict[key]
    else: return

def assert_nonan(tensor):
    assert th.any(th.isnan(tensor)).item() is False

def rslice(vid,coords):
    if coords is None: return vid
    if len(coords) == 0: return vid
    if th.is_tensor(coords):
        coords = coords.type(th.int)
        coords = list(coords.cpu().numpy())
    fs,fe,t,l,b,r = coords
    return vid[...,fs:fe,:,t:b,l:r]

def slice_flows(flows,t_start,t_end):
    if flows is None: return flows
    flows_t = edict()
    flows_t.fflow = flows.fflow[t_start:t_end]
    flows_t.bflow = flows.bflow[t_start:t_end]
    return flows_t

def write_pickle(fn,obj):
    with open(str(fn),"wb") as f:
        pickle.dump(obj,f)

def read_pickle(fn):
    with open(str(fn),"rb") as f:
        obj = pickle.load(f)
    return obj

def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)
    # th.random.
    # th.random.set_rng_state(seed)

def read_yaml(fn):
    with open(fn,"r") as stream:
        data = yaml.safe_load(stream)
    return data

# -- used for adaptation --
def get_region_gt(vshape):

    t,c,h,w = vshape
    hsize = min(h//4,128)
    wsize = min(w//4,128)
    tsize = min(t//4,5)

    t_start = max(t//2 - tsize//2,0)
    t_end = min(t_start + tsize,t)
    if t == 3: t_start = 0

    h_start = max(h//2 - hsize//2,0)
    h_end = min(h_start + hsize,h)

    w_start = max(w//2 - wsize//2,0)
    w_end = min(w_start + wsize,w)

    region_gt = [t_start,t_end,h_start,w_start,h_end,w_end]
    return region_gt

def nice_pretrained_path(pandas_series):
    split = pandas_series.str.split("-epoch=",expand=True)
    uuids = split[0]
    uuids_abbr = uuids.str.slice(0,4)
    epoch_num = split[1].str.slice(0,2)
    abbr = uuids_abbr + "-" + epoch_num
    return abbr
