
# -- imports --
from easydict import EasyDict as edict
from .space import space_chunks,extract_space_config
from .time import time_chunks,extract_time_config
from .channel import channel_chunks,extract_channel_config

# -- config --
from functools import partial
from ..common import optional as _optional
from ..common import optional_fields,extract_config
_fields = []
optional_full = partial(optional_fields,_fields)
extract_chunks_config = partial(extract_config,_fields)

#
# -- api --
#


# -- wrap only fwd fxn --
def wrap(cfg,model):
    model.forward = chunk(cfg,model.forward)

# -- main chunking --
def chunk(cfg,model):

    # -- allows for all keys to be aggregated at init --
    init = _optional(cfg,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)

    # -- unpack configs --
    channel_cfg = extract_channel_config(cfg,optional)
    space_cfg = extract_space_config(cfg,optional)
    time_cfg = extract_time_config(cfg,optional)
    if init: return

    # -- chunking --
    model_fwd = lambda vid,flows: model(vid,flows=flows)
    model_fwd = channel_chunks(channel_cfg,model_fwd)
    model_fwd = space_chunks(space_cfg,model_fwd)
    model_fwd = time_chunks(time_cfg,model_fwd)
    return model_fwd

# -- run to populate "_fields" --
chunk(edict({"__init":True}),None)
