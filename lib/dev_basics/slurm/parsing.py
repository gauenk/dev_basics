

import argparse
from pathlib import Path
from easydict import EasyDict as edict

def process_parser():
    desc = 'This will equip the script to accept input args to launch slum programs',
    parser = argparse.ArgumentParser(
        prog = 'Parser which equips a script to be run by with the Python Slurm laucher',
        description = desc,
        epilog = 'Happy Hacking')
    parser.add_argument('--start',type=int)
    parser.add_argument('--end',type=int)
    parser.add_argument('--clear',action="store_true")
    args = parser.parse_args()
    return args

def launcher_parser():
    desc = "Launches python scripts equipped with 'slurm_parser' to accept arguments",
    parser = argparse.ArgumentParser(
        prog = 'Launching Slurm Experiments',
        description = desc,
        epilog = 'Happy Hacking')
    parser.add_argument('script')
    parser.add_argument('total_exps',type=int)
    parser.add_argument('chunk_size',type=int)
    parser.add_argument('-J','--job_name_base',default=None)
    parser.add_argument('-c','--clear_first',action="store_true")
    parser.add_argument('-A','--account',default="standby")
    parser.add_argument('-M','--machines',nargs='+',default=["b","d"])
    parser.add_argument('-n','--nodes',default=1)
    parser.add_argument('-t','--time',default="0-4:00:00")
    parser.add_argument('--gpus_per_node',default=1)
    parser.add_argument('--cpus_per_task',default=2)
    parser.add_argument('--reset',action="store_true",
                        help="Clear out the slurm launch and output paths.")
    args = parser.parse_args()

    # -- fill default job name --
    if args.job_name_base is None:
        args.job_name_base = Path(args.script).stem

    return edict(vars(args))


