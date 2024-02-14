
# -- processing --
import torch as th
from easydict import EasyDict as edict
from .shared import get_chunks,expand_match,get_outputs
from ..common import _vprint
from functools import partial

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__,1)
extract_time_config = econfig.extract_config

#
# -- api --
#

# -- config --
def time_pairs():
    pairs = {"temporal_chunk_size":0,
             "temporal_chunk_overlap":0,
             "temporal_chunk_verbose":False}
    return pairs

# -- wrapper --
@econfig.set_init
def time_chunks(cfg,in_fwd):

    # -- extract --
    econfig.set_cfg(cfg)
    cfg = econfig({"time":time_pairs()}).time
    if econfig.is_init: return

    # -- unpack --
    size = cfg.temporal_chunk_size
    overlap = cfg.temporal_chunk_overlap
    verbose = cfg.temporal_chunk_verbose

    # -- run --
    out_fwd = in_fwd
    if not(size is None) and not(size == "none") and not(size <= 0):
        out_fwd = lambda vid,flows=None: run_temporal_chunks(in_fwd,size,overlap,vid,
                                                             flows=flows,verbose=verbose)
    return out_fwd

#
# -- meat --
#


def run_temporal_chunks(fwd_fxn,tsize,overlap,vid,flows=None,verbose=True):
    """
    overlap is a __percent__
    """

    # -- setup --
    vprint = partial(_vprint,verbose)
    nframes = vid.shape[-4]
    t_chunks = get_chunks(nframes,tsize,overlap)
    vprint("t_chunks: ",t_chunks)

    # -- output shape --
    C = vid.shape[-3]
    Cout = 3 if C in [3,4] else C
    Cout = C
    oshape = list(vid.shape)
    oshape[-3] = Cout

    # -- alloc --
    deno = None
    rH,rW = None,None
    # deno = th.zeros(oshape,device=vid.device)
    Z = th.zeros(nframes,device=vid.device)

    # -- run --
    for t_chunk in t_chunks:

        # -- extract --
        t_slice = slice(t_chunk,t_chunk+tsize)
        vid_chunk = vid[...,t_slice,:,:,:]
        vprint("t_chunk: ",t_chunk,vid_chunk.shape)
        flow_chunk = get_temporal_chunk_flow(flows,t_slice)

        # -- process --
        if flows: deno_chunk = fwd_fxn(vid_chunk,flow_chunk)
        else: deno_chunk = fwd_fxn(vid_chunk,flow_chunk)

        # -- outputs --
        deno,_,rH,rW = get_outputs(deno,Z,rH,rW,deno_chunk,vid_chunk,vid)

        # -- accumulate --
        nf,Cout = deno_chunk.shape[-4:-2]
        # print(deno.shape,deno_chunk.shape,vid.shape,t_slice)
        deno[...,t_chunk:t_chunk+nf,:Cout,:,:] += deno_chunk
        Z[t_slice] += 1

    # -- normalize --
    Z = expand_match(deno.shape,Z,-4)
    deno /= Z

    return deno

#
# -- mics --
#

def get_temporal_chunk_flow(flows,t_slice):
    if flows is None:
        return None
    out_flows = edict()
    out_flows.fflow = flows.fflow[...,t_slice,:,:,:].contiguous().clone()
    out_flows.bflow = flows.bflow[...,t_slice,:,:,:].contiguous().clone()

    # -- endpoints --
    out_flows.fflow[...,-1,:,:,:] = 0.
    out_flows.bflow[...,0,:,:,:] = 0.

    return out_flows


