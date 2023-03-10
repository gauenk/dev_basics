
# -- misc --
import copy
dcopy = copy.deepcopy
import numpy as np
from easydict import EasyDict as edict

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__,3) # init extraction
extract_config = econfig.extract_config # rename extraction


@econfig.set_init
def fill_blocks(cfg,blocks,blocklists,dfill):

    # -- arguments --
    econfig.init(cfg)
    cfg = econfig.optional_pairs(cfg,{"block_mlp":"mlp","norm_layer":"LayerNorm"})
    if econfig.is_init: return

    # -- fill from blocklists --
    fill_blocks_from_blocklists(blocks,blocklists,dfill)

    # -- fill common varaibles
    for block in blocks:
        for key,val in cfg.items():
            block[key] = val

    return blocks

def fill_blocks_from_blocklists(blocks,blocklists,fill_pydict):
    """

    Expand from a config to a list of configs
    with len(block) == # of blocks in network

    -=-=-=- Logical Abstractions -=-=-=-
    blocklist_0 -> blocklist_1 -> ....
    block_0,block_1,... -> block_0,block_1,... ->
    <----  depth_0 ---->   <---- depth_1 ---->

    -=-=-=- This Output -=-=-=-
    block_0,block_1,......,block_D0+1,block_D0+2,...
    <---- depth_0 -------><------- depth_1 -------->

    """
    start,stop = 0,0
    for blocklist in blocklists:
        start = stop
        stop = start + blocklist.depth
        for b in range(start,stop):
            block = blocks[b]
            for field,fill_fields in fill_pydict.items():
                if not(field in block):
                    block[field] = {}
                for fill_field in fill_fields:
                    if not(fill_field in block):
                        block[field][fill_field] = {}
                    block[field][fill_field] = blocklist[fill_field]

def copy_cfgs(cfgs,blocks,overwrite=False):
    for block in blocks:
        for name,cfg in cfgs.items():
            if not(name in block): block[name] = edict()
            for field,default in cfg:
                if field in block[name] and not(overwrite): continue
                block[name][field] = default
