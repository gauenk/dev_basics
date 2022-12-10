
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

def _vprint(verbose,*args,**kwargs):
    if verbose:
        print(*args,**kwargs)


