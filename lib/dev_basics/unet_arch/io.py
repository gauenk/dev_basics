

# -- misc --
from easydict import EasyDict as edict

# -- local --
# from dev_basics.unet_arch import menus as menus_lib
# from dev_basics.unet_arch import blocks as blocks_lib
# from dev_basics.unet_arch import blocklists as blocklists_lib

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__,2) # init extraction
extract_config = econfig.extract_config

# -- load the model --
@econfig.set_init
def load_arch(cfg,fill_cfgs,dfill=None):


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #          Arguments
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- defaults --
    if dfill is None:
        dfill = {"attn":["nheads","embed_dim"],"search":["nheads"],
                 "res":["nres_per_block","res_ksize"]}
    lib_pairs = {"menus_lib":"dev_basics.unet_arch.menus",
                 "blocks_lib":"dev_basics.unet_arch.blocks",
                 "blocklists_lib":"dev_basics.unet_arch.blocklists"}


    # -- unpack libs --
    econfig.init(cfg)
    libs = edict()
    econfigs = edict()
    for lib_field,lib_str in lib_pairs.items():
        name = lib_field.split("_")[0]
        lib = econfig.optional_module(cfg,lib_field,lib_str)
        libs[name] = lib
        econfigs[name] = lib.econfig

    # -- unpack configs --
    cfgs = econfig.optional_config_dict(cfg,econfigs,new=True)
    if econfig.is_init: return
    print("hey.")
    print(econfig.pairs)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #          Logic
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- load blocklists --
    blocklists = libs.blocklists.load_blocklists(cfgs.blocklists)

    # -- load blocks --
    blocks = libs.menus.load_blocks(cfgs.menus,fill_cfgs) # use menu
    blocks = libs.blocks.fill_blocks(cfgs.blocks,blocks,blocklists,dfill)

    return blocks,blocklists
