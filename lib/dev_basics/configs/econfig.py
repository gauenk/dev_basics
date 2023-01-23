

from easydict import EasyDict as edict

from functools import partial
from ..common import optional as _optional
from .impl import optional_fields
from .impl import extract_config,extract_pairs,cfg2lists
from .impl import SpoofModule
import importlib

class ExtractConfig():

    def __init__(self,fpath):
        self.fpath = fpath
        self.cfg = None
        self.fields = []
        self.pairs = {}
        self.is_init = False
        self.set_fxn = None
        self.init_key = '__init'
        self.init_fxn = None
        self.is_set = False

    def set_cfg(self,cfg):
        self.is_set = True
        self.cfg = cfg
        self.is_init = _optional(cfg,self.init_key,False) # purposefully weird key
        if self.is_init: # reset fields if init
            self.fields = []
            self.pairs = {}
        self.optional = partial(optional_fields,self.fields,self.pairs,self.is_init)

    def extract_config(self,_cfg,fill_defaults=True):
        self.init_fxn(edict({self.init_key:True} | _cfg))
        cfg = edict(extract_config(self.fields,_cfg))
        if fill_defaults:
            cfg = extract_pairs(self.pairs,cfg,_optional)
        return cfg

    def __call__(self,dicts_of_pairs):
        return self.extract(dicts_of_pairs)

    def extract(self,dicts_of_pairs):
        cfgs = edict()
        for cfg_name,pairs in dicts_of_pairs.items():
            cfg = extract_pairs(pairs,self.cfg,self.optional)
            cfgs[cfg_name] = cfg
        return cfgs

    def cfgs_to_lists(self,cfgs,keys,fixed_len):
        for key in keys:
            cfgs[key] = cfg2lists(cfgs[key],fixed_len)

    def set_init(self,fxn):
        self.init_fxn = fxn
        # if not(self.is_set):
        #     self.init_fxn({self.init_key:True})
        return fxn

    def optional_module(self,cfg,module_field):
        mname = self.optional(cfg,module_field,None)
        if not(mname is None):
            return importlib.import_module(mname)
        else:
            return SpoofModule()

    def required_module(self,cfg,module_field):
        if not(module_field in cfg):
            raise ValueError("Must have module field [%s]" % module_field)
        module = importlib.import_module(cfg[module_field])
        return module
