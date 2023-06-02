"""

Benchmark Forward/Backward Runtime/Memory

"""

# -- imports --
import numpy as np
import torch as th

# -- summary --
from torchinfo import summary as th_summary
from functools import partial
from easydict import EasyDict as edict

# -- model io --
import importlib

# -- data --
import data_hub

# -- optical flow --
from dev_basics import flow

# -- bench --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt


def run(cfg):

    # -- load --
    vid = load_sample(cfg)
    model = load_model(cfg)

    return run_loaded(model,vid,run_flows=cfg.flow,with_flows=True)

def run_loaded(model,vid,run_flows=False,with_flows=False):

    # -- timer/memer --
    timer = ExpTimer()
    memer = GpuMemer()

    # -- run flows --
    flows = flow.orun(vid,False) # init
    with TimeIt(timer,"flows"):
        with MemIt(memer,"flows"):
            flows = flow.orun(vid,run_flows)

    # -- def forward --
    def forward(vid):
        if with_flows:
            return model(vid,flows=flows)
        else:
            return model(vid)

    # -- init cuda --
    forward(vid)

    # -- bench fwd --
    with th.no_grad():
        with TimeIt(timer,"fwd_nograd"):
            with MemIt(memer,"fwd_nograd"):
                forward(vid)

    # -- bench fwd --
    with TimeIt(timer,"fwd"):
        with MemIt(memer,"fwd"):
            forward(vid)

    # -- compute grad --
    output = forward(vid)
    error = th.randn_like(output)
    loss = th.mean((error - output)**2)

    # -- bench fwd --
    with TimeIt(timer,"bwd"):
        with MemIt(memer,"bwd"):
            loss.backward()

    # -- fill results --
    results = edict()
    for key,val in timer.items():
        results[key] = val
    for key,(res,alloc) in memer.items():
        results["res_%s"%key] = res
        results["alloc_%s"%key] = alloc

    return results

def run_fwd(cfg):

    # -- timer/memer --
    timer = ExpTimer()
    memer = GpuMemer()

    # -- load --
    vid = load_sample(cfg)
    model = load_model(cfg)
    flows = flow.orun(vid,cfg.flow)
    vid = vid[0]

    # -- init --
    model(vid,flows=flows)

    # -- bench fwd --
    with th.no_grad():
        with TimeIt(timer,"fwd_nograd"):
            with MemIt(memer,"fwd_nograd"):
                model(vid,flows=flows)
    # -- fill results --
    results = {}
    for key,val in timer.items():
        results[key] = val
    for key,(res,alloc) in memer.items():
        results["res_%s"%key] = res
        results["alloc_%s"%key] = alloc

    return results

def load_model(cfg):
    return importlib.import_module(cfg.python_module).load_model(cfg)

def load_sample(cfg):
    # -- init data --
    device = "cuda:0"
    imax = 255.
    data,loaders = data_hub.sets.load(cfg)
    dset = "tr" if not("dset" in cfg) else cfg['dset']
    if "vid_name" in cfg:
        indices = data_hub.filter_nframes(data[dset],cfg.vid_name,
                                          cfg.frame_start,cfg.nframes)
    else:
        indices = [0]
    sample = data[dset][indices[0]]
    return sample['noisy'][None,:].to(device)

def print_summary(cfg,vshape,with_flows=True):
    model = load_model(cfg)
    flows = edict()
    fshape = (vshape[0],vshape[1],2,vshape[-2],vshape[-1])
    flows.fflow = th.randn(fshape).to("cuda:0")
    flows.bflow = th.randn(fshape).to("cuda:0")
    if with_flows:
        model.forward = partial(model.forward,flows=flows)
    th_summary(model, input_size=vshape)

def summary(cfg,vshape,with_flows=True):
    model = load_model(cfg)
    return summary_loaded(model,vshape,with_flows)

def summary_loaded(model,vshape,with_flows=True):

    # -- fwd/bwd times&mem --
    vid = th.randn(vshape).to("cuda:0")
    res = run_loaded(model,vid,run_flows=False,with_flows=with_flows)

    # -- view&append summary --
    if with_flows:
        flows = edict()
        fshape = (vshape[0],vshape[1],2,vshape[-2],vshape[-1])
        flows.fflow = th.randn(fshape).to("cuda:0")
        flows.bflow = th.randn(fshape).to("cuda:0")
        model.forward = partial(model.forward,flows=flows)
    summ = th_summary(model, input_size=vshape, verbose=0)
    res.total_params = summ.total_params / 1e6
    res.trainable_params = summ.trainable_params / 1e6
    res.macs = summ.total_mult_adds / 1e9
    res.fwdbwd_mem = summ.total_output_bytes / 1e9

    res.total_params = sum(p.numel() for p in model.parameters())/ 1e6
    res.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    res.trainable_params = res.trainable_params/1e6

    return res


