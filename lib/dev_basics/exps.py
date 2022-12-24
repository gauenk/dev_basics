"""

Manage experiment files

"""

import yaml
from .mesh import mesh_groups,add_cfg

def load(fn): # read + unpack
    edata = read(fn)
    return unpack(edata)

def read(fn):
    with open(fn,"r") as stream:
        data = yaml.safe_load(stream)
    return data

def unpack(edata):
    cfg = edata['cfg']
    groups = [v for g,v in edata.items() if "group" in g]
    grids = [v for g,v in edata.items() if "global_grids" in g]
    exps = []
    for grid in grids:
        exps += mesh_groups(grid,groups)
    add_cfg(exps,cfg)
    return exps

