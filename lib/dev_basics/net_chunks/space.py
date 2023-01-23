
# -- processing --
import torch as th
from .shared import get_chunks
from easydict import EasyDict as edict
from ..common import _vprint
from functools import partial

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__,1)
extract_space_config = econfig.extract_config

#
# -- api --
#

# -- config --
def space_pairs():
    pairs = {"spatial_chunk_size":0,
             "spatial_chunk_overlap":0,
             "spatial_chunk_verbose":False}
    return pairs

# -- exposed --
@econfig.set_init
def space_chunks(cfg,in_fwd):

    # -- extract --
    econfig.set_cfg(cfg)
    cfg = econfig({"space":space_pairs()}).space
    if econfig.is_init: return

    # -- unpack --
    size = cfg.spatial_chunk_size
    overlap = cfg.spatial_chunk_overlap
    verbose = cfg.spatial_chunk_verbose
    out_fwd = in_fwd

    # -- run --
    if not(size is None) and not(size == "none") and not(size <= 0):
        out_fwd = lambda vid,flows: run_spatial_chunks(in_fwd,size,overlap,vid,
                                                       flows=flows,verbose=verbose)
    return out_fwd

#
# -- meat --
#

def run_spatial_chunks(fwd_fxn,size,overlap,vid,flows=None,verbose=False):
    """
    overlap is a _percent_

    """

    # -- setup --
    vprint = partial(_vprint,verbose)
    H,W = vid.shape[-2:] # .... H, W

    # -- get chunks --
    h_chunks = get_chunks(H,size,overlap)
    w_chunks = get_chunks(W,size,overlap)

    # -- alloc --
    deno = th.zeros_like(vid)
    Z = th.zeros((H,W),device=vid.device)

    # -- run --
    vprint("h_chunks,w_chunks: ",h_chunks,w_chunks)
    for h_chunk in h_chunks:
        for w_chunk in w_chunks:
            vid_chunk = get_spatial_chunk(vid,h_chunk,w_chunk,size)
            flow_chunk = get_spatial_chunk_flow(flows,h_chunk,w_chunk,size)
            vprint("s_chunk: ",h_chunk,w_chunk,vid_chunk.shape)
            if flows: deno_chunk = fwd_fxn(vid_chunk,flow_chunk)
            else: deno_chunk = fwd_fxn(vid_chunk,flow_chunk)
            fill_spatial_chunk(deno,deno_chunk,h_chunk,w_chunk,size)
            fill_spatial_chunk(Z,1,h_chunk,w_chunk,size)

    # -- normalize --
    eshape = (1,) * len(deno.shape[:-2])
    eshape += deno.shape[-2:]
    Z = Z.expand(eshape)
    deno /= Z # normalize across overlaps
    return deno

#
# -- mics --
#

def get_spatial_chunk(vid,h_chunk,w_chunk,size):
    return vid[...,h_chunk:h_chunk+size,w_chunk:w_chunk+size]

def fill_spatial_chunk(vid,ivid,h_chunk,w_chunk,size):
    vid[...,h_chunk:h_chunk+size,w_chunk:w_chunk+size] += ivid

def get_spatial_chunk_flow(flows,h_chunk,w_chunk,size):

    # -- allow none --
    if flows is None: return None

    # -- new dict --
    out_flows = edict()
    out_flows.fflow = flows.fflow[...,h_chunk:h_chunk+size,w_chunk:w_chunk+size]
    out_flows.bflow = flows.bflow[...,h_chunk:h_chunk+size,w_chunk:w_chunk+size]

    # -- contig --
    out_flows.fflow = out_flows.fflow.contiguous().clone()
    out_flows.bflow = out_flows.bflow.contiguous().clone()

    return out_flows

