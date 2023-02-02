"""

Primary interface for saving denoised examples.

"""

# -- non-internal imports --
from pathlib import Path
from easydict import EasyDict as edict

# -- imports --
import cache_io # possible circular import... danger but I like this better 02/01/23

from dev_basics.utils.misc import read_yaml
from .impl import save_example,filter_df


def run(example_cfg,results):

    # -- load deno examples info --
    cfg = edict(read_yaml(example_cfg))
    save_args = cfg.save_args
    labels = cfg.labels
    save_args.root = Path(save_args.root)
    examples = [cfg[key] for key in cfg.keys() if 'example' in key]

    # -- global filter --
    results = filter_df(results,cfg.global_filter)

    # -- save each example --
    paths,missing = [],[]
    for example in examples:
        ex_paths,ex_missing = save_example(results,example,save_args,labels)
        paths.append(ex_paths)
        missing.append(ex_missing)

    # -- return --
    return paths,missing

def get_results(example_cfg,exp_fxn):

    # -- load/run experiments. --
    cfg = edict(read_yaml(example_cfg))
    info = cfg.cache_info # must exist

    # -- load/run cache --
    results_fn = info.results_fn
    results_reload = info.results_reload
    name,version = info.name,info.version
    skip_loop = info.skip_loop
    exp_files = info.exps
    exps = cache_io.get_exps(exp_files)
    results = cache_io.run_exps(exps,exp_fxn,name=name,version=version,
                                skip_loop=skip_loop,results_fn=results_fn,
                                results_reload=results_reload)
    assert len(results) >= len(exps),"Must have all experiments complete."
    return results
