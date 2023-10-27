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
import socket

hostname = socket.gethostname()
if "anvil" in hostname:
    default_account = "gpu-debug"
    default_time = "0-0:10:00"
    default_gpus = 1
else:
    default_account = "standby"
    default_time = "0-4:00:00"
    default_gpus = 1

def parse():
    desc = 'Launch a single python command using sbatch',
    parser = argparse.ArgumentParser(
        prog = 'Commandline launch a single command',
        description = desc,
        epilog = 'Happy Hacking')
    parser.add_argument("launch_cmd",type=str,
                        help="The command to launch with sbatch")
    parser.add_argument("-A","--account",type=str,default=default_account,
                        help="The account to launch")
    parser.add_argument("-T","--time",type=str,default=default_time,
                        help="The time limit of the proc")
    parser.add_argument("--ncpus",type=int,default=4,
                        help="The number of cpus to launch.")
    parser.add_argument("--ngpus",type=int,default=default_gpus,
                        help="The number of gpus to launch.")
    parser.add_argument("--job_name",type=str,default="sbatch_cmd",
                        help="The name of the job.")
    parser.add_argument("-C",type=str,default="A100|a100",
                        help="Constraints")
    parser.add_argument("-O","--outfile",type=str,default="sbatch_cmd.txt",
                        help="The output file.")
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
    print("\n\nCheck %s for info." % args.outfile)

def get_tmpdir():
    cmd = ["echo $TMPDIR"]
    proc = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    tmpdir = proc.stdout.strip()
    return tmpdir

def create_content(args,cwd):
    cmd = "#!/bin/sh -l\n"
    if "anvil" in hostname:
        cmd += "#SBATCH -p %s\n" % args.account
    else:
        cmd += "#SBATCH -A %s\n" % args.account
    if args.C != "":
        cmd += "#SBATCH -C %s\n" % args.C
    cmd += "#SBATCH --nodes 1\n"
    cmd += "#SBATCH --time %s\n" % args.time
    cmd += "#SBATCH --cpus-per-task %d\n" % args.ncpus
    if args.ngpus > 0:
        cmd += "#SBATCH --gpus-per-node %d\n" % args.ngpus
    cmd += "#SBATCH --job-name %s\n" % args.job_name
    cmd += "#SBATCH --output %s\n" % args.outfile
    cmd += "WORKDIR=%s\n" % cwd
    cmd += "/bin/hostname\n"
    cmd += args.launch_cmd
    return cmd

