
# -- misc --
import copy
dcopy = copy.deepcopy
import numpy as np
from easydict import EasyDict as edict

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__) # init extraction
extract_config = econfig.extract_config # rename extraction


@econfig.set_init
def load_blocklists(cfg):

    # -- arguments --
    econfig.init(cfg)
    info = {"mlp_ratio":4.,"embed_dim":1,"block_version":"v3",
            "freeze":False,"block_mlp":"mlp","norm_layer":"LayerNorm",
            "num_res":3,"res_ksize":3,"nres_per_block":3,
            "depth":[1,1,1],"nheads":[1,1,1]}
    training = {"drop_rate_mlp":0.,"drop_rate_path":0.1}
    pairs = info | training
    cfg = econfig.optional_pairs(cfg,pairs)
    if econfig.is_init: return

    # -- create blocklists --
    cfg.nblocklists = 2*(len(cfg.depth)-1)+1
    blocklists = init_blocklists(cfg,cfg.nblocklists)
    set_channels(blocklists)

    return blocklists

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Expand Varaible Length Lists to # of Blocklists
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def init_blocklists(cfg,L):
    """

    Expands dicts with field of length 1, 1/2, or Full length
    lists into a list of dicts

    """
    # converts a edict to a list of edicts
    cfgs = []
    keys = list(cfg.keys())
    for l in range(L):
        cfg_l = edict()
        for key in keys:
            if isinstance(cfg[key],list):
                mid = L//2
                eq = len(cfg[key]) == L
                eq_h = len(cfg[key]) == (mid+1)
                assert eq or eq_h,"Must be shaped for %s & %d" % (key,L)
                if eq: # index along the list
                    cfg_l[key] = cfg[key][l]
                elif eq_h: # reflect list length is half size
                    li = l if l <= mid else ((L-1)-l)
                    cfg_l[key] = cfg[key][li]
            else:
                cfg_l[key] = cfg[key]
        cfgs.append(cfg_l)
    return cfgs

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Create Up/Down Scales
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def set_channels(blocklists):
    # -- create scales for each blocklist --
    fill_downsample_cfg(blocklists)
    fill_upsample_cfg(blocklists)

def fill_upsample_cfg(bcfgs):
    # cfgs = []
    start = len(bcfgs)//2-1
    for l in range(start,0-1,-1):
        l_p = (len(bcfgs)-1) - l
        cfg_l = bcfgs[l_p]#edict()
        cfg_l.in_dim = bcfgs[l+1].embed_dim*bcfgs[l+1].nheads
        if l != start:
            cfg_l.in_dim = 2 * cfg_l.in_dim
        cfg_l.out_dim = bcfgs[l].embed_dim*bcfgs[l].nheads
        # cfgs.append(cfg_l)

    # -- center is None
    bcfgs[start+1].in_dim = None
    bcfgs[start+1].out_dim = None

def fill_downsample_cfg(bcfgs):
    nencs = len(bcfgs)//2
    for l in range(nencs):
        cfg_l = bcfgs[l]#edict()
        cfg_l.in_dim = bcfgs[l].embed_dim*bcfgs[l].nheads
        cfg_l.out_dim = bcfgs[l+1].embed_dim*bcfgs[l+1].nheads
        # cfgs.append(cfg_l)

