
"""

A commandline utility for launching slurm files

slurm_launcher ./script/my_python_script <num_experiments> <experiments_per_process>

"""

import os
from pathlib import Path
from .api import run

def main():
    base = Path(os.getcwd()).resolve() / "slurm"
    run(base)

if __name__ == "__main__":
    main()
