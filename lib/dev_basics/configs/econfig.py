"""

Usage:

# .. imports here..

# -- config import --
from dev_basics.config import ExtractConfig
econfig = ExtractConfig(__file__)
extract_config = econfig.extract_config

@econfig.set_init
def my_fxn_with_configs(dict_like_object):

  pairs0 = {"param1":param1_default,
           "param2":param2_default,...}
  pairs1 = {"param1":param1_default,
           "param2":param2_default,...}
  econfig.set_cfg(dict_like_object)
  cfg = econfig({"0":pairs0,"1":pairs1})
  if econfig.is_init: return

  # .. rest of function ...

Functionality:

Returns a completed dictionary

Example:

# < other script >

import some_func

cfg = {"param_a":100}
cfg_func = some_func.extract_config(cfg)
# cfg_func = {"param_a":100,"param_b":20,"param_c":30}


"""

import copy
dcopy = copy.deepcopy
from easydict import EasyDict as edict

from functools import partial
from ..common import optional as _optional
from .impl import optional_append
from .impl import extract_pairs,copy_cfg_fields,cfg2lists
from .impl import SpoofModule
import importlib

class ExtractConfig():

    def __init__(self,fpath,nargs=0):
        self.fpath = fpath
        self.nargs = nargs
        # self.cfg = None
        self.pairs = {}
        self.is_init = False
        self.set_fxn = None
        self.init_key = '__init'
        self.init_fxn = None
        self.is_set = False

    def set_cfg(self,cfg):
        self.init(cfg)

    def init(self,cfg):
        self.is_set = True
        self.cfg = cfg
        self.is_init = _optional(cfg,self.init_key,False) # purposefully weird key
        if self.is_init: # reset if init
            self.pairs = edict()
        self.optional = partial(optional_append,self.pairs,self.is_init)
        # return dcopy(cfg)

    def optional_field(self,cfg,field,default):
        return self.optional(cfg,field,default)

    def optional_pairs(self,cfg,pairs,new=True):
        if new: out_cfg = edict()
        else: out_cfg = cfg
        for field,default in pairs.items():
            out_cfg[field] = self.optional(cfg,field,default)
        return out_cfg

    def optional_config(self,cfg,econfig,new=True):
        # -- 1.) unpack config --
        cfg = econfig.extract_config(cfg,new=new)
        # -- 2.) append to pairs --
        for key,val in econfig.pairs.items():
            self.pairs[key] = val
        return cfg

    def optional_config_dict(self,cfg,econfig_dict,new=True):
        cfgs = edict()
        for name,econfig in econfig_dict.items():
            cfgs[name] = self.optional_config(dcopy(cfg),econfig,new=new)
        return cfgs

    def extract_config(self,cfg,new=True): # external api
        """

        Returns _all_ and _only_ (new=True) the fields used to call the function
        decorated with "set_init"

        The default pairs are generated dynamically when called.

        Allow input _cfg to dynamically set subsequent configs.

        """

        # -- generate required fields --
        self.run_init(cfg)

        # -- optionally create new config --
        if new:
            cfg = edict(copy_cfg_fields(list(self.pairs.keys()),cfg))
        else:
            cfg = _cfg

        # -- primary logic --
        cfg = extract_pairs(cfg,self.pairs,_optional)

        # -- idk why it appears --
        if self.init_key in cfg:
            del cfg[self.init_key]
        return cfg

    def extract_dict(self,cfg,extract_dict): # internal api
        return self.extract_dict(cfg,extract_dict)

    def extract_dict_of_econfigs(self,cfg,extract_dict): # internal api
        cfgs = edict()
        for name,econfig in extract_dict.items():
            cfgs[name] = econfig.extract_config(cfg,new=True)
            # -- 2.) append to pairs --
            for key,val in econfig.pairs.items():
                self.pairs[key] = val
        return cfgs

    def extract_list(self,cfg,extract_list): # internal api
        for econfig in extract_list:
            cfg = econfig.extract_fxn(cfg,new=False)
        # self.pairs = merge_pairs([v.pairs for v in extract_list])
        return cfg

    def extract_pairs(self,_cfg,pairs,new=True,restrict=False): # internal api
        return extract_pairs(_cfg,pairs,self.optional,new=new,restrict=restrict)

    def extract_set(self,dicts_of_pairs,new=True): # internal api
        return self.extract_dict_of_pairs(self.cfg,dicts_of_pairs,new=new)

    # internal api
    def extract_dict_of_pairs(self,cfg,dicts_of_pairs,new=True,restrict=False):
        # raise NotImplemented("Don't use this.")
        cfgs = edict()
        for cfg_name,pairs in dicts_of_pairs.items():
            _cfg = self.extract_pairs(cfg,pairs,new=new,restrict=restrict)
            cfgs[cfg_name] = _cfg
        return cfgs

    def __call__(self,dicts_of_pairs):
        return self.extract_set(dicts_of_pairs)

    def flatten(self,cfgs):
        cfg = edict()
        for cfg_i in cfgs.values():
            cfg = edict({**cfg,**cfg_i})
        return cfg

    def cfgs_to_lists(self,cfgs,keys,fixed_len):
        for key in keys:
            cfgs[key] = cfg2lists(cfgs[key],fixed_len)

    def cfgs2lists(self,cfg,fixed_len):
        return cfg2lists(cfg,fixed_len)

    def set_init(self,fxn):
        self.init_fxn = fxn
        return fxn

    def run_init(self,_cfg):
        cfg = dict(_cfg)
        cfg.update({self.init_key:True})
        cfg = edict(cfg)
        if self.nargs > 0:
            args = self.nargs*[None,]
            self.init_fxn(cfg,*args)
        else:
            self.init_fxn(cfg)

    def optional_module(self,cfg,module_field,module_default=None):
        mname = self.optional(cfg,module_field,module_default)
        if mname is None:
            return SpoofModule()
        else:
            return importlib.import_module(mname)

    def required_module(self,cfg,module_field):
        if not(module_field in cfg):
            raise ValueError("Must have module field [%s]" % module_field)
        module = importlib.import_module(cfg[module_field])
        return module
