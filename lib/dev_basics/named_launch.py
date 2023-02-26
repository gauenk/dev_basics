"""

Relaunch a process when it is not running or in queue on slurm.

Works only with the "sbatch_py" command since the relaunch occurs from "dispatch" folder

named_launch <job_id> <job_uuid> <njobs> <njobs_per_proc> <user_id>

"""

import os
import tqdm
import time
import argparse
import subprocess
import datetime
from pathlib import Path
from easydict import EasyDict as edict

def parse():
    desc = 'Relaunch the sbatch_py processes and check every X minutes',
    parser = argparse.ArgumentParser(
        prog = 'Allows us to relaunch sbatch procs by name',
        description = desc,
        epilog = 'Happy Hacking')
    parser.add_argument("job_id",type=str,
                        help="The shared id of the jobs.")
    parser.add_argument("job_uuid",type=str,
                        help="The uuid of the jobs from sbatch_py.")
    parser.add_argument("njobs",type=int,
                        help="The total number of jobs")
    parser.add_argument("njobs_per_proc",type=int,
                        help="The number of jobs per process.")
    parser.add_argument("user_id",type=str,default="gauenk",
                        help="The account name for squeue")
    parser.add_argument('--interval',type=float,default=1.,
                        help="Wait [interval] minutes. Then check for relaunching.")
    parser.add_argument('--min_time',type=int,default=2,
                        help="Minimum wait time in minutes")
    args = parser.parse_args()
    return edict(vars(args))

def main():

    # -- init --
    print("PID: ",os.getpid())
    args = parse()

    # -- setup paths --
    launch_files = get_launch_files(args)
    job_names = get_job_names(args)
    job_times = {n:timedelta_from_timefield("1:0:0") for n in job_names}
    min_time = datetime.timedelta(minutes=args.min_time)
    viewed = {n:False for n in job_names}

    # -- check each process --
    while True:
        running_names,running_times = get_running_info(args)
        # print(job_times,flush=True)
        # print(running_times)
        for name,fn in zip(job_names,launch_files):
            if name in running_names:
                index = running_names.index(name)
                job_times[name] = running_times[index]
                continue
            if job_times[name] <= min_time: # job is complete.
                if viewed[name] is False:
                    print("Job name %s is complete." % name,flush=True)
                    viewed[name] = True
                continue
            # if viewed[name] is True: continue # job completed but relaunched?
            print("Relaunching job name %s" % name,flush=True)
            run_launch_file(fn)
            # set to be marked as complete if it complete before time is updated.
            job_times[name] = datetime.timedelta(seconds=0)
        sleep_bar(args.interval)

def get_running_info(args):
    cmd = ["squeue","-u",args.user_id]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    slurm_info = proc.stdout
    jobs = slurm_info.split("\n")[1:]
    job_names,job_times = [],[]
    for job in jobs:
        job_args = job.split()
        if len(job_args) != 9:
            continue
        job_names.append(job_args[3])
        job_times.append(timedelta_from_timefield(job_args[8]))
    return job_names,job_times

def timedelta_from_timefield(time_str):
    times = reversed(time_str.split(":"))
    fields = ["seconds","minutes","hours"]
    kwargs = {f:int(t) for f,t in zip(fields,times)}
    dtime = datetime.timedelta(**kwargs)
    return dtime

def run_launch_file(fn):
    cmd = ["sbatch",fn]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    slurm_info = proc.stdout

def get_job_names(args):
    names = []
    nprocs = (args.njobs-1) // args.njobs_per_proc + 1
    for proc_i in range(nprocs):
        job_start = proc_i * args.njobs_per_proc
        name_i = "%s_%d" % (args.job_id,job_start)
        names.append(name_i)
    return names

def get_launch_files(args):
    launch_dir = Path("./dispatch/%s/launching/" % args.job_id)
    launch_files = []
    nprocs = (args.njobs-1) // args.njobs_per_proc + 1
    for proc_i in range(nprocs):
        job_start = proc_i * args.njobs_per_proc
        job_end = job_start + args.njobs_per_proc
        fn_i = "%s_%d_%d.sh" % (args.job_uuid,job_start,job_end)
        launch_files.append(launch_dir / fn_i)
    return launch_files

def sleep_bar(nmins):
    # for i in tqdm.tqdm(range(nmins),desc="Minutes until next check."):
    time.sleep(int(60.*nmins))
    

