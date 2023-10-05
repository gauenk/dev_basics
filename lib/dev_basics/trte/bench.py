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
from dev_basics import net_chunks

# -- model io --
import importlib

# -- data --
import data_hub

# -- optical flow --
from dev_basics import flow

# -- bench --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from dev_basics.utils.misc import optional

def run(cfg):

    # -- load --
    fwd_only = optional(cfg,'bench_fwd_only',False)
    vid = load_sample(cfg)
    model = load_model(cfg).to("cuda")
    chunk_fxn = partial(net_chunks.chunk,cfg)
    return run_loaded(model,vid,run_flows=cfg.flow,
                      with_flows=True,chunk_fxn=chunk_fxn,fwd_only=fwd_only)

def run_loaded(model,vid,run_flows=False,with_flows=False,fwd_only=False,chunk_fxn=None):

    # -- timer/memer --
    timer = ExpTimer()
    memer = GpuMemer()
    if fwd_only: model.eval()

    # -- run flows --
    flows = flow.orun(vid,False) # init
    with TimeIt(timer,"flows"):
        with MemIt(memer,"flows"):
            flows = flow.orun(vid,run_flows)

    # -- def forward --
    def forward(vid):
        fwd_fxn = chunk_fxn(model.forward)
        if with_flows:
            return fwd_fxn(vid,flows=flows)
        else:
            return fwd_fxn(vid)

    # -- init cuda --
    forward(smaller_vid(vid))

    # -- bench fwd --
    with th.no_grad():
        with TimeIt(timer,"fwd_nograd"):
            with MemIt(memer,"fwd_nograd"):
                forward(vid)

    if fwd_only is False:

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

def smaller_vid(vid):
    B,T,C,H,W = vid.shape
    return vid[:1,:7,:,:256,:256]

def run_fwd(cfg):

    # -- timer/memer --
    timer = ExpTimer()
    memer = GpuMemer()

    # -- load --
    vid = load_sample(cfg)
    model = load_model(cfg).to("cuda")
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

def run_fwd_vshape(model,vshape,run_flows=False,with_flows=False):

    # -- timer/memer --
    timer = ExpTimer()
    memer = GpuMemer()

    # -- get video --
    vid = th.randn(vshape).to("cuda:0")

    # -- run flows --
    flows = flow.orun(vid,False) # init
    with TimeIt(timer,"flows"):
        with MemIt(memer,"flows"):
            flows = flow.orun(vid,run_flows)

    # -- def forward --
    def forward(_vid):
        if with_flows:
            return model(_vid,flows=flows)
        else:
            return model(_vid)

    # -- init cuda --
    forward(vid[:1,:3,:,:256,:256])

    # -- bench fwd --
    with th.no_grad():
        with TimeIt(timer,"fwd_nograd"):
            with MemIt(memer,"fwd_nograd"):
                forward(vid)

    # -- fill results --
    results = edict()
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
    vid = sample['noisy'][None,:].to(device)
    if "dd_in" in cfg and cfg.dd_in == 4:
        noise = th.zeros_like(vid[:,:,:1])
        vid = th.cat([vid,noise],-3)
    return vid

def print_summary(cfg,vshape,with_flows=True):
    model = load_model(cfg).to("cuda")
    flows = edict()
    fshape = (vshape[0],vshape[1],2,vshape[-2],vshape[-1])
    flows.fflow = th.randn(fshape).to("cuda:0")
    flows.bflow = th.randn(fshape).to("cuda:0")
    if with_flows:
        model.forward = partial(model.forward,flows=flows)
    th_summary(model, input_size=vshape)

def summary(cfg,vshape,with_flows=True,fwd_only=False):
    model = load_model(cfg).to("cuda")
    chunk_fxn = partial(net_chunks.chunk,cfg)

    return summary_loaded(model,vshape,with_flows,fwd_only,chunk_fxn)

def wrap_flows(model,vshape):
    flows = edict()
    fshape = (vshape[0],vshape[1],2,vshape[-2],vshape[-1])
    flows.fflow = th.randn(fshape).to("cuda:0")
    flows.bflow = th.randn(fshape).to("cuda:0")
    return partial(model.forward,flows=flows)
    # model.forward = partial(model.forward,flows=flows)

def summary_loaded(model,vshape,with_flows=True,fwd_only=False,chunk_fxn=None):

    # -- fwd/bwd times&mem --
    vid = th.randn(vshape).to("cuda:0")
    res = run_loaded(model,vid,run_flows=False,with_flows=with_flows,chunk_fxn=chunk_fxn)

    # -- view&append summary --
    if with_flows:
        model.forward = wrap_flows(model,vshape)
    summ = th_summary(model, input_size=vshape, verbose=0)
    res.total_params = summ.total_params / 1e6
    # res.trainable_params = summ.trainable_params / 1e6
    res.macs = summ.total_mult_adds / 1e9
    res.fwdbwd_mem = summ.total_output_bytes / 1e9
    res.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    res.trainable_params = res.trainable_params/1e6

    res.total_params = sum(p.numel() for p in model.parameters())/ 1e6
    res.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    res.trainable_params = res.trainable_params/1e6

    return res


