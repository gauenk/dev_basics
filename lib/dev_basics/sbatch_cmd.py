"""

Run a single command using sbatch

"""

import os
import tqdm
import time
import argparse
import subprocess
from pathlib import Path
from easydict import EasyDict as edict

def parse():
    desc = 'Launch a single python command using sbatch',
    parser = argparse.ArgumentParser(
        prog = 'Commandline launch a single command',
        description = desc,
        epilog = 'Happy Hacking')
    parser.add_argument("launch_cmd",type=str,
                        help="The command to launch with sbatch")
    parser.add_argument("--account",type=str,default="standby",
                        help="The account to launch")
    parser.add_argument("--time",type=str,default="0-4:00:00",
                        help="The time limit of the proc")
    parser.add_argument("--ncpus",type=int,default=1,
                        help="The number of cpus to launch.")
    parser.add_argument("--ngpus",type=int,default=1,
                        help="The number of gpus to launch.")
    args = parser.parse_args()
    return edict(vars(args))

def main():

    # -- init --
    args = parse()
    tmpdir = Path(get_tmpdir())
    cwd = os.getcwd()
    sbatch_sh = create_content(args,cwd)
    tmpfile = tmpdir / "sbatch_cmd.sh"

    # -- write file --
    with open(str(tmpfile),"w") as f:
        f.write(sbatch_sh)

    # -- launch command --
    print("[Launching] sbatch_py %s" % str(tmpfile))
    cmd = ["sbatch",str(tmpfile)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    slurm_stdout = proc.stdout
    slurm_stderr = proc.stderr
    print("---> Stdout <---")
    print(slurm_stdout)
    print("---> Stderr <---")
    print(slurm_stderr)
    print("\n\nCheck sbatch_cmd.txt for info.")

def get_tmpdir():
    cmd = ["echo $TMPDIR"]
    proc = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    tmpdir = proc.stdout.strip()
    return tmpdir

def create_content(args,cwd):
    cmd = "#!/bin/sh -l\n"
    cmd += "#SBATCH -A %s\n" % args.account
    cmd += "#SBATCH --nodes 1\n"
    cmd += "#SBATCH --time %s\n" % args.time
    cmd += "#SBATCH --cpus-per-task %d\n" % args.ncpus
    if args.ngpus > 0:
        cmd += "#SBATCH --gpus-per-node %d\n" % args.ngpus
    cmd += "#SBATCH --job-name sbatch_cmd\n"
    cmd += "#SBATCH --output sbatch_cmd.txt\n"
    cmd += "WORKDIR=%s\n" % cwd
    cmd += "/bin/hostname\n"
    cmd += args.launch_cmd
    return cmd

