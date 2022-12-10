
# -- processing --
import torch as th
from easydict import EasyDict as edict
from .shared import get_chunks,expand_match

# -- configs --
from functools import partial
from ..common import extract_pairs,_vprint

#
# -- api --
#

# -- config --
def extract_channel_config(_cfg,optional):
    pairs = {"channel_chunk_size":0,
             "channel_chunk_overlap":0,
             "channel_chunk_verbose":False}
    return extract_pairs(pairs,_cfg,optional)

# -- wrapper --
def channel_chunks(cfg,in_fwd):
    size = cfg.channel_chunk_size
    overlap = cfg.channel_chunk_overlap
    verbose = cfg.channel_chunk_verbose
    out_fwd = in_fwd
    if not(size is None) and not(size == "none") and not(size <= 0):
        out_fwd = lambda vid,flows: run_channel_chunks(in_fwd,size,overlap,vid,
                                                       flows=flows,verbose=verbose)
    return out_fwd

#
# -- meat --
#

def run_channel_chunks(fwd_fxn,size,overlap,vid,flows=None,verbose=False):
    """
    overlap is a _percent_

    """
    # -- init --
    vprint = partial(_vprint,verbose)
    C = vid.shape[-3] # .... C, H, W

    # -- alloc --
    deno = th.zeros_like(vid)
    Z = th.zeros((C,1,1),device=vid.device)

    # -- run --
    c_chunks = get_chunks(C,size,overlap)
    vprint("c_chunks: ",c_chunks)
    for c_chunk in c_chunks:
        vid_chunk = get_channel_chunk(vid,c_chunk,size)
        vprint("s_chunk: ",c_chunk,vid_chunk.shape)
        deno_chunk = fwd_fxn(vid_chunk,flows)
        fill_channel_chunk(deno,deno_chunk,c_chunk,size)
        fill_channel_chunk(Z,1,c_chunk,size)

    # -- normalize --
    V = len(vid.shape)
    Z = Z.expand((1,)*(V-3) + Z.shape)
    # Z = expand_match(deno.shape,Z,-3)
    deno /= Z
    return deno

#
# -- misc --
#

def get_channel_chunk(vid,c_chunk,size):
    return vid[...,c_chunk:c_chunk+size,:,:]

def fill_channel_chunk(vid,chunk,c_chunk,size):
    vid[...,c_chunk:c_chunk+size,:,:] += chunk

