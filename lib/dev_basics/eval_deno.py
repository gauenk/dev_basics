"""

Commandline compute psnrs between shared folders of two directories

eval_deno <path_to_deno> <path_to_clean>

"""

import os
import tqdm
import time
import argparse
import subprocess
import datetime
from pathlib import Path
from easydict import EasyDict as edict
from dev_basics import vid_io
from dev_basics import metrics
import numpy as np

def parse():
    desc = 'Compute PSNRS',
    parser = argparse.ArgumentParser(
        prog = 'Compute psnrs via commandline',
        description = desc,
        epilog = 'Happy Hacking')
    parser.add_argument("deno_root",type=str,
                        help="Root of denoised folders")
    parser.add_argument("clean_root",type=str,
                        help="Root of clean folders")
    args = parser.parse_args()
    return edict(vars(args))

def main():

    # -- init --
    print("PID: ",os.getpid())
    args = parse()
    deno_root = Path(args.deno_root)
    clean_root = Path(args.clean_root)

    # -- setup paths --
    names,psnrs,ssims = [],[],[]
    deno_vids = list(deno_root.iterdir())
    for deno_dir in tqdm.tqdm(deno_vids):

        # -- check clean exists --
        clean_dir = clean_root / deno_dir.name
        if not(clean_dir.exists()):
            print(f"Skipping video [{deno_dir.name}] since clean DNE.")
            continue

        # -- read --
        deno = vid_io.read_video(deno_dir)
        clean = vid_io.read_video(clean_dir)

        # -- compute metrics --
        vid_name = deno_dir.name
        psnrs_vid = metrics.compute_psnrs(deno,clean)
        ssims_vid = metrics.compute_ssims(deno,clean)

        # -- append --
        names.append(vid_name)
        psnrs.append(psnrs_vid.mean())
        ssims.append(ssims_vid.mean())

    # -- viz --
    print(names)
    print(psnrs)
    print(np.mean(psnrs))
    print(ssims)
    print(np.mean(ssims))

