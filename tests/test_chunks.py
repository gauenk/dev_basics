
# -- python --
import sys

# -- data mgnmt --
from pathlib import Path
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- dev_basics --
from dev_basics import net_chunks

#
# -- Primary Testing Class --
#

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    vshape_list = {"T":[6,8],"C":[3,5,9],"H":[32,128],"W":[64,96]}
    chunk_list = {"spatial_chunk_size":[32,64],
                   "spatial_chunk_overlap":[0,0.25],
                   "temporal_chunk_size":[0,2,4],
                   "temporal_chunk_overlap":[0,0.3],
                   "channel_chunk_size":[0,3,9],
                   "channel_chunk_overlap":[0,0.5]}
    test_list = vshape_list | chunk_list
    for key,val in test_list.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test_chunks(T,C,H,W,spatial_chunk_size,spatial_chunk_overlap,
                temporal_chunk_size,temporal_chunk_overlap,
                channel_chunk_size,channel_chunk_overlap):

    # -- test config --
    device = "cuda:0"
    B = 3
    vshape = (B,T,C,H,W)

    # -- chunk config --
    cfg = edict()
    cfg.spatial_chunk_size = spatial_chunk_size
    cfg.spatial_chunk_overlap = spatial_chunk_overlap
    cfg.temporal_chunk_size = temporal_chunk_size
    cfg.temporal_chunk_overlap = temporal_chunk_overlap
    cfg.channel_chunk_size = channel_chunk_size
    cfg.channel_chunk_overlap = channel_chunk_overlap

    # -- vid --
    vid = th.ones(vshape,device=device)
    sizes = []
    def model_fwd(vid,**kwargs):
        sizes.append(np.array(list(vid.shape)))
        return vid

    # -- chunks --
    chunk_cfg = net_chunks.extract_chunks_config(cfg)
    model_fwd = net_chunks.chunk(chunk_cfg,model_fwd)
    out = model_fwd(vid,None)

    # -- compare --
    diff = th.mean((out - vid)**2).item()
    assert diff < 1e-10

    # -- inspect num of runs --
    nruns = net_chunks.expected_runs(cfg,T,C,H,W)
    # assert len(sizes) - nruns == 0

    # -- inspect sizes --
    t_size = cfg.temporal_chunk_size
    c_size = cfg.channel_chunk_size
    s_size = cfg.spatial_chunk_size
    t_size = min(t_size,T) if t_size > 0 else T
    c_size = min(c_size,C) if c_size > 0 else C
    sh_size = min(s_size,H) if s_size > 0 else H
    sw_size = min(s_size,W) if s_size > 0 else W
    tgt_size = np.array([B,t_size,c_size,sh_size,sw_size])
    error = 0
    for size in sizes:
        error += np.mean((tgt_size - size)**2*1.)
    assert error < 1e-10
