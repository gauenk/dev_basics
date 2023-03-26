"""

The number of layers ~= 50
This is too many params to set by hand.
Instead, fix the list of parameters using a menu.

"""

# -- misc --
import copy
dcopy = copy.deepcopy
import numpy as np
from easydict import EasyDict as edict

# -- local --
# from . import menus
from . import search

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__,1) # init extraction
extract_config = econfig.extract_config # rename extraction

@econfig.set_init
def load_blocks(cfg,fill_cfgs):

    # -- config --
    econfig.init(cfg)

    # -- unpack attn name --
    # ...

    # -- unpack search name --
    # "search_vX" in ["exact","refine","approx_t","approx_s","approx_st"]
    search_cfg = econfig.optional_config(cfg,search.econfig,new=True)

    # -- unpack normz name --
    normz_blocks = [{} for _ in search_cfg]

    # -- unpack agg name --
    agg_blocks = [{} for _ in search_cfg]

    # -- misc --
    depth = econfig.optional_field(cfg,'depth',[1,1,1])

    # -- finish args --
    if econfig.is_init: return

    # -- comp --
    nblocks = 2*np.sum(depth[:-1]) + depth[-1]
    search_blocks = search.load_blocks(search_cfg)

    # -- combine blocks --
    blocks = []
    block_types = {"search":search_blocks,
                   "normz":normz_blocks,
                   "agg":agg_blocks}
    for i in range(nblocks):
        block_i = edict()
        for name,block_type in block_types.items():
            if not(name in block_i): block_i[name] = edict()
            for key,val in block_type[i].items():
                block_i[name][key] = val
        blocks.append(block_i)

    # -- fill with defaults --
    fill_menu(fill_cfgs,blocks)

    return blocks


def fill_menu(fill_cfgs,blocks,overwrite=True):
    """

    Fill the input "_cfgs" fields using a menu.

    """

    # -- filling --
    for block in blocks:
        for block_field,field_cfg in fill_cfgs.items():
            if not(block_field in block):
                block[block_field] = edict()
            for field,value in field_cfg.items():
                if field in block[block_field] and not(overwrite):
                    continue
                block[block_field][field] = value

