"""

Some basic IO for dev

"""

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
    if wtype in ["git","original"]:
        load_checkpoint_git(model,path,root)
    elif wtype in ["lightning","lit"]:
        load_checkpoint_lit(model,path,root)
    elif wtype in ["mod"]:
        load_checkpoint_mod(model,path,root,mod)
    else:
        raise ValueError(f"Uknown checkpoint weight type [{wtype}]")

def load_checkpoint_lit(model,path,root):
    # -- filename --
    if not Path(path).exists():
        path = str(Path(root) / "output/checkpoints/" / Path(path))
    assert Path(path).exists()
    weights = th.load(path)
    state = weights['state_dict']
    remove_lightning_load_state(state)
    model.load_state_dict(state)

def load_checkpoint_git(model,path,root):
    # -- filename --
    if not Path(path).exists():
        path = str(Path(root) / "output/checkpoints/" / Path(path))
    checkpoint = th.load(path)
    model.load_state_dict(checkpoint)

def load_checkpoint_mod(model,path,root,modifier):
    # -- filename --
    if not Path(path).exists():
        path = str(Path(root) / "output/checkpoints/" / Path(path))
    state = th.load(path)['state_dict']
    modifier(state)
    model.load_state_dict(state)
