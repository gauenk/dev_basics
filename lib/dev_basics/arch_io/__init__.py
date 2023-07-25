"""

Some basic IO for dev

"""

import pickle
import torch as th
from pathlib import Path

def remove_lightning_load_state(state):
    names = list(state.keys())
    for name in names:
        name_new = name.split(".")[1:]
        name_new = ".".join(name_new)
        state[name_new] = state[name]
        del state[name]

def load_checkpoint(model, path, root, wtype="git", mod=None):
    path = resolve_path(path,root)
    if wtype in ["git","original"]:
        load_checkpoint_git(model,path)
    elif wtype in ["npy"]:
        load_checkpoint_npy(model,path)
    elif wtype in ["lightning","lit"]:
        load_checkpoint_lit(model,path)
    elif wtype in ["litr"]:
        load_checkpoint_litr(model,path)
    elif wtype in ["mod"]:
        load_checkpoint_mod(model,path,mod)
    else:
        raise ValueError(f"Uknown checkpoint weight type [{wtype}]")

# def resolve_path(path,root):
#     if file_exists(path):
#         path_ = Path(root) / Path(path)
#         path = resolve_cycle(original_path)
#         # if not(path_.exists()):
#         #     path_ = Path(root) / "output/checkpoints/" / Path(path)
#         # path = path_
#     assert Path(path).exists()
#     return str(path)

def resolve_path(path,root):

    # -- pathlib paths --
    path_s = str(path)
    path = Path(path)
    root = Path(root)

    # -- 0.) check input --
    v0 = Path(root) / Path(path)
    # print(v0)
    exists = file_exists(v0)
    if exists: return v0

    # -- 1.) check "output/checkpoints/" --
    v1 = Path(root) / "output/checkpoints/" / Path(path)
    exists = file_exists(v1)
    if exists: return v1

    # -- 2.) check "_uuid_here-epoch..." --
    uuid = get_uuid(path_s)
    v2 = Path(root) / uuid / Path(path)
    exists = file_exists(v2)
    if exists: return v2

    # -- 3.) check "_uuid_here-epoch..." --
    v3 = Path(root) / "output/train/checkpoints" / uuid / Path(path)
    exists = file_exists(v3)
    if exists: return v3

    # -- 4.) check "_uuid_here-epoch..." --
    v4 = Path(root) / "checkpoints" / uuid / Path(path)
    exists = file_exists(v4)
    if exists: return v4
    # print(v4)

    # -- error out --
    msg = "Uknown checkpoint path. Failed to load checkpoint.\n%s\n%s" % (root,path)
    raise ValueError(msg)

def get_uuid(path_s):
    uuid = path_s[:path_s.find("-epoch")]
    if "save" in uuid:
        uuid = uuid[:uuid.find("-save")]
    return uuid

def file_exists(path):
    return path.exists() and path.is_file()

def load_checkpoint_lit(model,path):
    # -- filename --
    weights = th.load(path)
    state = weights['state_dict']
    remove_lightning_load_state(state)
    model.load_state_dict(state)

def load_checkpoint_litr(model,path):
    # -- filename --
    weights = th.load(path)
    state = weights['state_dict']
    remove_lightning_load_state(state)
    model.load_state_dict(state,strict=False)

def load_checkpoint_git(model,path):
    # -- filename --
    checkpoint = th.load(path)
    model.load_state_dict(checkpoint)

def load_checkpoint_npy(model,path):
    # -- filename --
    # with open(path, 'rb') as f:
    #     checkpoint = pickle.load(f)
    # checkpoint = checkpoint['model']
    load_numpy_state_dict(model,path)

def load_checkpoint_mod(model,path,modifier):
    # -- filename --
    state = th.load(path)
    state = modifier(state)
    model.load_state_dict(state)

# load from numpy
def load_numpy_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')['model']

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(th.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))

