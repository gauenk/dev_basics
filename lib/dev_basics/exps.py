"""

Manage experiment files

probably belongs in cache_io

"""

import yaml
from .mesh import mesh_groups,add_cfg

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

