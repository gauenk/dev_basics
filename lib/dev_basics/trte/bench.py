"""

Benchmark Forward/Backward Runtime/Memory

"""

# -- imports --
import numpy as np
import torch as th

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

    # -- bench fwd --
    with TimeIt(timer,"fwd"):
        with MemIt(memer,"fwd"):
            model(vid,flows=flows)

    # -- compute grad --
    deno = model(vid,flows=flows)
    error = th.randn_like(deno)
    loss = th.mean((error - deno)**2)

    # -- bench fwd --
    with TimeIt(timer,"bwd"):
        with MemIt(memer,"bwd"):
            loss.backward()

    # -- fill results --
    results = {}
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
    imax = 255.
    data,loaders = data_hub.sets.load(cfg)
    indices = data_hub.filter_nframes(data[cfg.dset],cfg.vid_name,
                                      cfg.frame_start,cfg.nframes)
    sample = data[cfg.dset][indices[0]]
    return sample['noisy'][None,:].to(cfg.device)

