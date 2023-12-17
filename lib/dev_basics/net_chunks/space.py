
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
        out_fwd = lambda vid,flows=None: run_spatial_chunks(in_fwd,size,overlap,vid,
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
    C = vid.shape[-3]

    # -- output shape --
    # Cout = 3 if C in [3,4] else C
    Cout = C
    oshape = list(vid.shape)
    oshape[-3] = Cout

    # -- get chunks --
    h_chunks = get_chunks(H,size,overlap)
    w_chunks = get_chunks(W,size,overlap)

    # -- alloc --
    deno = th.zeros(oshape,device=vid.device)
    Z = th.zeros((H,W),device=vid.device)
    # deno,Z = None,None
    # rH,rW = None,None

    # -- run --
    vprint("h_chunks,w_chunks: ",h_chunks,w_chunks)
    for h_chunk in h_chunks:
        for w_chunk in w_chunks:
            vid_chunk = get_spatial_chunk(vid,h_chunk,w_chunk,size)
            flow_chunk = get_spatial_chunk_flow(flows,h_chunk,w_chunk,size)
            vprint("s_chunk: ",h_chunk,w_chunk,vid_chunk.shape)
            if flows: deno_chunk = fwd_fxn(vid_chunk,flow_chunk)
            else: deno_chunk = fwd_fxn(vid_chunk,flow_chunk)
            # deno,Z,rH,rW = get_outputs(deno,Z,rH,rW,deno_chunk,vid_chunk,vid)
            Cout,sizeH,sizeW = deno_chunk.shape[-3:]
            fill_spatial_chunk(deno,deno_chunk,h_chunk,w_chunk,sizeH,sizeW)
            fill_spatial_chunk(Z,1,h_chunk,w_chunk,sizeH,sizeW)
            # fill_spatial_chunk(deno,deno_chunk,rH*h_chunk,rW*w_chunk)
            # fill_spatial_chunk(Z,1,rH*h_chunk,rW*w_chunk)

    # -- normalize --
    eshape = (1,) * len(deno.shape[:-2])
    eshape += deno.shape[-2:]
    Z = Z.expand(eshape)
    deno /= Z # normalize across overlaps
    return deno

#
# -- mics --
#

def get_outputs(deno,Z,rH,rW,vid_out_chunk,vid_in_chunk,vid_in_full):
    if deno is None:
        return allocate_outputs(vid_out_chunk,vid_in_chunk,vid_in_full)
    else:
        return deno,Z,rH,rW

def allocate_outputs(vid_out_chunk,vid_in_chunk,vid_in_full):
    """

    Allocate output shape depending on the network's output shape.

    """
    device = vid_out_chunk.device
    H,W,rH,rW = get_output_spatial_size(vid_out_chunk,vid_in_chunk,vid_in_full)
    oshape = list(vid_out_chunk.shape[:-2]) + [H,W]
    deno = th.zeros(oshape,device=device)
    Z = th.zeros((H,W),device=device)
    return deno,Z,rH,rW

def get_output_spatial_size(vid_out_chunk,vid_in_chunk,vid_in_full):
    outH,outW = vid_out_chunk.shape[-2:]
    inH,inW = vid_in_chunk.shape[-2:]
    rH,rW = int(outH/(1.*inH)),int(outW/(1.*inW))
    H,W = vid_in_full.shape[-2:]
    outH,outW = rH*H,rW*W
    return outH,outW,rH,rW

def get_spatial_chunk(vid,h_chunk,w_chunk,size):
    return vid[...,h_chunk:h_chunk+size,w_chunk:w_chunk+size]

def fill_spatial_chunk(vid,ivid,h_chunk,w_chunk,sizeH,sizeW):
    # sizeH,sizeW = vid.shape[-2:]
    # print(vid.shape,ivid.shape)
    # vid[...,h_chunk:h_chunk+size,w_chunk:w_chunk+size] += ivid
    vid[...,h_chunk:h_chunk+sizeH,w_chunk:w_chunk+sizeW] += ivid

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

