"""

Manage experiment files

"""

import yaml
from .mesh import mesh_groups,add_cfg

def read(fn):
    with open(fn,"r") as stream:
        data = yaml.safe_load(stream)
    return data

def unpack(edata):
    cfg = edata['cfg']
    grids = edata['global_grids']
    groups = [v for g,v in edata.items() if "group" in g]
    exps = mesh_groups(grids,groups)
    add_cfg(exps,cfg)
    return exps

