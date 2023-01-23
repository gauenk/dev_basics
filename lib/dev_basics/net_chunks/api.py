
# -- imports --
from easydict import EasyDict as edict
from .space import space_chunks,extract_space_config
from .time import time_chunks,extract_time_config
from .channel import channel_chunks,extract_channel_config

# -- config --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__,1)
extract_chunks_config = econfig.extract_config

#
# -- api --
#


# -- wrap only fwd fxn --
def wrap(cfg,model):
    model.forward = chunk(cfg,model.forward)

# -- main chunking --
@econfig.set_init
def chunk(cfg,model):

    # -- unpack configs --
    econfig.set_cfg(cfg)
    cfgs = econfig({"channel":extract_channel_config(cfg),
                    "space":extract_space_config(cfg),
                    "time":extract_time_config(cfg)})
    if econfig.is_init: return

    # -- chunking --
    model_fwd = lambda vid,flows: model(vid,flows=flows)
    model_fwd = channel_chunks(cfgs.channel,model_fwd)
    model_fwd = space_chunks(cfgs.space,model_fwd)
    model_fwd = time_chunks(cfgs.time,model_fwd)
    return model_fwd
