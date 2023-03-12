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
from dev_basics.utils.misc import transpose_dict_list

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__) # init extraction
extract_config = econfig.extract_config # rename extraction

@econfig.set_init
def load_blocks(cfg):

    # -- start of parsing --
    econfig.init(cfg)

    # -- extract --
    pairs = {"search_menu_name":"full",
             "search_v0":"exact",
             "search_v1":"refine",
             "depth":[1,1,1],
             "nheads":[1,1,1]}
    cfg = econfig.extract_pairs(cfg,pairs)

    # -- end of parsing --
    if econfig.is_init: return
    print("unet_arch.menus.search: ",cfg)

    # -- init --
    params = edict()
    params.search_names = get_search_names(cfg.search_menu_name,cfg.depth,
                                           cfg.search_v0,cfg.search_v1)
    params.use_state_updates = get_use_state_updates(params.search_names)

    # -- transpose --
    blocks = transpose_dict_list(params)

    return blocks

def search_menu(depth,menu_name,v0,v1):

    # -- init --
    params = edict()
    params.search_names = get_search_names(menu_name,depth,v0,v1)
    params.use_state_updates = get_use_state_updates(params.search_names)
    return params

def get_use_state_updates(search_names):
    """
    Create derived parameters from parsed parameters

    """
    # -- fill --
    nblocks = len(search_names)
    any_refine = np.any(np.array(search_names)=="refine")
    use_state_updates = []
    for i in range(nblocks):
        use_state_updates.append(any_refine)
    return use_state_updates

def get_search_names(menu_name,depth,v0,v1):
    nblocks = 2*np.sum(depth[:-1]) + depth[-1]

    if menu_name == "full":
        return [v0,]*nblocks
    elif menu_name == "one":
        return [v0,] + [v1,]*(nblocks-1)
    elif menu_name == "first":
        names = []
        for depth_i in depth:
            names_i = [v0,] + [v1,]*(depth_i-1)
            names.extend(names_i)
        for depth_i in reversed(depth[:-1]):
            names_i = [v0,] + [v1,]*(depth_i-1)
            names.extend(names_i)
        return names
    elif menu_name == "nth":
        names = []
        for i in range(nblocks):
            if (i % menu_n == 0) or i == 0:
                names.append(v0)
            else:
                names.append(v1)
        return names
    else:
        raise ValueError("Uknown search type in menu [%s]" % menu_name)

def fill_menu(_cfgs,fields,menu_cfgs,mfields):
    """

    Fill the input "_cfgs" fields using a menu.

    """

    # -- filling --
    cfgs = []
    for menu_cfg in menu_cfgs:
        cfgs_m = edict()
        for field in fields:
            cfg_f = dcopy(_cfgs[field])
            for fill_key in mfields[field]:
                cfg_f[fill_key] = menu_cfg[fill_key]
            cfgs_m[field] = cfg_f
        cfgs.append(cfgs_m)
    return cfgs

