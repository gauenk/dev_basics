
# -- python --
import sys

# -- data mgnmt --
from pathlib import Path
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- dev_basics --
from dev_basics import net_chunks

# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

#
# -- Primary Testing Class --
#

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    vshape_list = {"T":[6,8],"C":[3,5,9],"H":[32,128],"W":[64,96]}
    test_list = vshape_list
    for key,val in test_list.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test_flows(T,C,H,W):
    pass
