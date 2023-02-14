
# -- imports --
from easydict import EasyDict as edict

# -- optionally access dict --
def optional(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default

def _vprint(verbose,*args,**kwargs):
    if verbose:
        print(*args,**kwargs)


#
#
#   ... TO REMOVE ...
#
#

# -- auto populate fields to extract config --
def optional_fields(_fields,init,pydict,field,default):
    # print("move to dev_basics.configs")
    if not(field in _fields) and init:
        _fields.append(field)
    return optional(pydict,field,default)

def extract_pairs(pairs,_cfg,optional):
    # print("move to dev_basics.configs")
    cfg = edict()
    for key,val in pairs.items():
        cfg[key] = optional(_cfg,key,val)
    return cfg

def extract_config(_fields,_cfg):
    # print("move to dev_basics.configs")
    cfg = {}
    for field in _fields:
        if field in _cfg:
            cfg[field] = _cfg[field]
    return edict(cfg)

def set_defaults(defs,pairs,overwrite=True):
    # print("move to dev_basics.configs")
    for key,val in defs.items():
        if overwrite:
            pairs[key] = val
        elif not(key in pairs):
            pairs[key] = val

def cfg2lists(cfg,L):
    # print("move to dev_basics.configs")
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
                assert eq or eq_h,"Must be shaped for %s & %s" % key
                if eq: # index along the list
                    cfg_l[key] = cfg[key][l]
                elif eq_h: # reflect list length is half size
                    li = l if l <= mid else ((L-1)-l)
                    cfg_l[key] = cfg[key][li]
            else:
                cfg_l[key] = cfg[key]
        cfgs.append(cfg_l)
    return cfgs
