"""

Saves denoised examples from the test_models.py script

"""

# -- misc --
import os
from pathlib import Path
from collections import OrderedDict

# -- wrangle --
import pandas as pd
import numpy as np
import torch as th

# -- yaml-io --
from dev_basics.utils.misc import read_yaml
from easydict import EasyDict as edict

# -- dev basics --
from dev_basics.utils import vid_io
from dev_basics.utils.metrics import compute_psnrs,compute_ssims,compute_strred

# -- images --
from PIL import Image
from torchvision.utils import make_grid
from einops import rearrange

# -- caching results --
import cache_io # possibly circular import. yikes! but I like this better.

# -- original data --
import data_hub # again, possibly circular import. yikes! but I like this better.

def filter_df(df,pydict):
    fskip = ["regions","data_hub_crop"]
    for field,val in pydict.items():
        if field in fskip: continue
        df = df[df[field] == val]
    return df

def extract_denos(vids,label,df_i):
    assert len(df_i) == 1
    row = df_i.reset_index(drop=True).iloc[0]
    fns = sorted(row['deno_fns'][0])
    vid = vid_io.read_files(fns)
    vids[label] = vid

def compute_metrics(clean,deno):
    m = edict()
    m.psnrs = compute_psnrs(clean,deno,255).mean()
    m.ssims = compute_ssims(clean,deno,255).mean()
    m.strred = compute_strred(clean[...,:128,:128],deno[...,:128,:128],255).mean()
    return m

def extract_metrics(agg,label,df_i):
    assert len(df_i) == 1
    row = df_i.reset_index(drop=True).iloc[0]
    metrics = ['psnrs','ssims','strred']
    agg_i = {}
    for metric in metrics:
        agg_i[metric] = np.mean(row[metric])
    agg[label] = agg_i

def run_extraction(df,labels,fxn):
    agg = OrderedDict({})
    for label,conditions in labels.items():
        df_i = filter_df(df,conditions)
        assert len(df_i) == 1,"Must be len = 1."
        fxn(agg,label,df_i)
    return agg

def load_pair(cfg,region):
    # -- load data --
    data,loaders = data_hub.sets.load(cfg)
    index = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,
                                   -1,-1)[0]
    clean = data[cfg.dset][index]['clean']
    noisy = data[cfg.dset][index]['noisy']

    # -- apply optional region --
    clean = apply_region(region,clean)
    noisy = apply_region(region,noisy)

    return noisy,clean

def build_base(cfg,index):
    return "%s_%s_%d" % (cfg.vid_name,cfg.sigma,index)

def apply_region(region,vid):
    # -- check if exists --
    if region is None: return vid

    # -- unpack --
    _region = []
    for r in region.split(","):
        if r == "null":
            _region.append(None)
        else:
            _region.append(int(r))
    fs,fe,hs,he,ws,we = _region

    # -- apply --
    return vid[fs:fe,:,hs:he,ws:we]

def get_region(region,vids):

    # -- filter --
    vids_c = OrderedDict({})
    for label in vids:
        vid_c = apply_region(region,vids[label])
        vids_c[label] = vid_c
    return vids_c

def vid_merge(*vids,nrow=6):
    vid = []
    for frames in zip(*vids):
        grid = make_grid(list(frames),nrow=nrow,pad_value=255)
        vid.append(grid)
    vid = th.stack(vid)
    return vid

def save_video_v1(root,vids,metrics):

    # -- info --
    print("Saving video & metrics at %s" % root)

    # -- unpack from Ordered Dict --
    labels,vids = zip(*vids.items())
    labels_c = [l.replace("\\","").replace("  "," ") for l in labels]

    # -- merge and save vid --
    vid = vid_merge(*vids)
    print("vid.shape: ",vid.shape)
    vid_io.save_video(vid,root,"frame")

    # -- save metrics --
    fn = root / "metrics.csv"
    metrics = pd.DataFrame(metrics).T
    metrics['strred'] = metrics['strred']*100
    fields = list(metrics.columns)
    metrics = metrics.rename_axis(index="Labels").reset_index()
    metrics = metrics[fields + ["Labels"]]
    metrics_c = metrics.copy()
    metrics_c["Labels"] = labels_c
    metrics_c.to_csv(fn,float_format='%2.2f',index=False)

    # -- save latex line --
    fn = root / "latex.txt"
    fmts = ["%2.2f","%0.3f","%2.2f"]
    info = create_latex_line(metrics,labels,fields,fmts)
    with open(fn,"w") as f:
        f.write(info)

def create_latex_line(metrics,labels,fields,fmts):
    tab_fmt = "\\begin{tabular}{@{}c@{}} %s \\end{tabular}"

    info = ""
    for label in labels:
        label_s = label
        if len(label_s) > 10 and "\\\\" in label_s:
            label_s = tab_fmt % label_s
        info += "%s "%(label_s)
        info += " & "
    info = info[:-2] + "\\\\" + "\n"
    for label in labels:
        if label in ["Clean"]:
            info += " & "
            continue
        row = metrics[metrics["Labels"] == label]
        _info = [fmt % row[field] for fmt,field in zip(fmts,fields)]
        info += "/".join(_info)[:-1]
        info += " & "
    info = info[:-2] + "\\\\" + "\n"
    return info

def save_video_v2(fn,vids):
    # -- [v1] only save deno --
    vids = [vids[o] for o in order]
    vid_io.save_video(fn,deno)

def save_video(grid_fmt,root,vids,metrics):
    if grid_fmt == "v1":
        save_video_v1(root,vids,metrics)
    else:
        raise ValueError(f"Uknown mode [{mode}]")

def save_example(df_full,example,save_args,labels):

    # -- get example data --
    df = filter_df(df_full,example)
    if len(df) == 0:
        vname,sigma = example['vid_name'],example['sigma']
        print("Missing results for expeirment [%s] @ [%s]" % (vname,sigma))
        return [],[vname,sigma]

    # -- load full denoised result from path --
    vids = run_extraction(df,labels,extract_denos)
    metrics = run_extraction(df,labels,extract_metrics)
    print(metrics)

    # -- get noisy/clean pair --
    noisy,clean = load_pair(example,example.data_hub_crop)
    vids['Noisy'] = noisy
    vids['Clean'] = clean
    metrics['Noisy'] = compute_metrics(clean,noisy)
    metrics['Clean'] = compute_metrics(clean,clean+.1)

    # -- order elements --
    order = save_args.order
    vids = OrderedDict({o:vids[o] for o in order})
    metrics = OrderedDict({o:metrics[o] for o in order})

    # -- save each region --
    roots = []
    for index,region in enumerate(example.regions):
        # -- save (video region,base_fn) --
        vids_r = get_region(region,vids)
        root = save_args.root / build_base(example,index)
        save_video(save_args.grid_format,root,vids_r,metrics)
        roots.append(root)

    return roots,[]
