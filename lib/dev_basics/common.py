
# -- imports --
from easydict import EasyDict as edict

# -- auto populate fields to extract config --
def optional_fields(_fields,init,pydict,field,default):
    if not(field in _fields) and init:
        _fields.append(field)
    return optional(pydict,field,default)

# -- optionally access dict --
def optional(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default

def extract_pairs(pairs,_cfg,optional):
    cfg = edict()
    for key,val in pairs.items():
        cfg[key] = optional(_cfg,key,val)
    return cfg

def extract_config(_fields,_cfg):
    cfg = {}
    for field in _fields:
        if field in _cfg:
            cfg[field] = _cfg[field]
    return edict(cfg)

def set_defaults(defs,pairs):
    for key,val in defs.items():
        pairs[key] = val

def _vprint(verbose,*args,**kwargs):
    if verbose:
        print(*args,**kwargs)

def cfg2lists(cfg,L):
    # converts a edict to a list of edicts
    cfgs = []
    keys = list(cfg.keys())
    for l in range(L):
        cfg_l = edict()
        for key in keys:
            if isinstance(cfg[key],list):
                mid = L//2
                eq = len(cfg[key]) == L
                eq_h = len(cfg[key]) == (mid+1)
                assert eq or eq_h
                if eq: # index along the list
                    cfg_l[key] = cfg[key][l]
                elif eq_h: # reflect list length is half size
                    li = l if l <= mid else ((L-1)-l)
                    cfg_l[key] = cfg[key][li]
            else:
                cfg_l[key] = cfg[key]
        cfgs.append(cfg_l)
    return cfgs
