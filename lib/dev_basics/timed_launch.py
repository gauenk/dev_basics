"""

Relaunch a process every X minutes

"""

import tqdm
import time
import argparse
import subprocess
from pathlib import Path
from easydict import EasyDict as edict

def parse():
    desc = 'Relaunch a program every X minutes',
    parser = argparse.ArgumentParser(
        prog = 'Allows us to relaunch killed slurm procs',
        description = desc,
        epilog = 'Happy Hacking')
    parser.add_argument('command',type=str)
    parser.add_argument('--time',type=int,default=245,
                        help="Wait 4 hours and 5 minutes. Then relaunch.")
    parser.add_argument('--skip_first',action="store_true",
                        help="Do we launch the first execution?")
    args = parser.parse_args()
    return edict(vars(args))

def main():
    args = parse()
    cmd = args.command
    # cmd_args = cmd.split(" ")
    first = True
    nmins = args.time
    while True: # run forever
        if not(first and args.skip_first): # allow skipping of first
            proc = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            slurm_info = proc.stdout
            print(slurm_info)
        else:
            print("skipping.")
        first = False
        sleep_bar(nmins)

def sleep_bar(nmins):
    for i in tqdm.tqdm(range(nmins),desc="Minutes until next launch."):
        time.sleep(60)
    
