"""

Manage experiment files

probably belongs in cache_io

"""

import yaml
from easydict import EasyDict as edict
from .mesh import mesh_groups,add_cfg
from .mesh import read_rm_picked,append_picked

def load(fn): # read + unpack
    print("[dev_basics.exps] Moving this to cache_io.")
    edata = read(fn)
    return unpack(edata)

def read(fn):
    print("[dev_basics.exps] Moving this to cache_io.")
    with open(fn,"r") as stream:
        data = yaml.safe_load(stream)
    return data

def unpack(edata):
    print("[dev_basics.exps] Moving this to cache_io.")
    cfg = edata['cfg']
    groups = [v for g,v in edata.items() if "group" in g]
    grids = [v for g,v in edata.items() if "global_grids" in g]
    exps = []
    for grid in grids:
        exps += mesh_groups(grid,groups)
    add_cfg(exps,cfg)
    return exps

def load_picked(fn):
    edata = read(fn)
    picks = read_rm_picked(edata)
    exps = unpack(edata)
    return append_picked(exps,picks)

def get_exps(exp_file_or_list):
    islist = isinstance(exp_file_or_list,list)
    ispath = isinstance(exp_file_or_list,edict)
    if islist:
        isdict = isinstance(exp_file_or_list[0],edict)
        isdict = isdict or isinstance(exp_file_or_list[0],dict)
        if isdict:
            exps = exp_file_or_list
        else: # list of config files
            exps = []
            for fn in exp_file_or_list:
                exps.extend(load(fn))
    else: # single list of config files
        exps = load(exp_file_or_list)
    return exps
