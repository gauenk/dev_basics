

from functools import partial
from ..common import optional as _optional
from .impl import optional_fields
from .impl import extract_config,extract_pairs,cfg2lists

class ExtractConfig():

    def __init__(self,fpath):
        self.fpath = fpath
        self.cfg = None
        self.fields = []
        self.is_init = False
        self.set_fxn = None
        self.init_key = '__init'
        self.init_fxn = None

    def set_cfg(self,cfg):
        self.cfg = cfg
        self.is_init = _optional(cfg,'__init',False) # purposefully weird key
        if self.is_init: self.fields = [] # reset fields if init
        self.optional = partial(optional_fields,self.fields,self.is_init)

    def extract_config(self,cfg):
        self.init_fxn({self.init_key:True})
        # assert self.is_set,"Must set [%s] before extracting configs." % self.fpath
        return extract_config(self.fields,cfg)

    def __call__(self,configs):
        return self.extract(configs)

    def extract(self,configs):
        cfgs = {}
        for cfg_name,defaults in configs.items():
            cfg = extract_pairs(pairs,self.cfg,self.optional)
            cfgs[cfg_name] = cfg
        return cfgs

    def cfgs_to_lists(self,cfgs,keys,fixed_len):
        for key in keys:
            cfgs[key] = cfg2lists(cfgs[key],fixed_len)

    def set_init(self,fxn):
        self.init_fxn = fxn
