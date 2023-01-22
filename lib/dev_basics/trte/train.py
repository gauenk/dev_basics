
# -- misc --
import os,copy
dcopy = copy.deepcopy

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- modules --
import importlib

# -- pytorch-lit --
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.utilities.distributed import rank_zero_only

# -- dev basics --
from .. import flow
from ..utils.misc import set_seed
from ..utils.misc import write_pickle
from ..utils.timer import ExpTimer,TimeIt
from ..utils.metrics import compute_psnrs,compute_ssims

# -- extract config --
from ..configs import ExtractConfig
econfig = ExtractConfig(__file__)
extract_config = ExtractConfig.extract_config

def train_pairs():
    pairs = {"num_workers":4,
             "persistent_workers":True,
             "rand_order_tr":True,
             "gradient_clip_algorithm":"norm",
             "index_skip_val":5,
             "root":".","seed":123,
    }
    return pairs

@econfig.set_init
def run(cfg):

    # -=-=-=-=-=-=-=-=-
    #
    #     Init Exp
    #
    # -=-=-=-=-=-=-=-=-

    # -- config --
    econfig.set_cfg(cfg):
    python_module = econfig.optional(cfg,"python_module",None)
    module = importlib.import_module(python_mod)
    ModelLit = module.lightning.LitModel
    lit_extract_config = module.lightning.extract_config
    net_extract_config = module.extract_config
    cfgs = econfig.extract_pairs({"tr":train_pairs(),
                                  "lit":lit.extract_config(cfg),
                                  "net":net.extract_config(cfg)})
    if cfgs.is_init: return

    # -- set-up --
    print("PID: ",os.getpid())
    set_seed(cfgs.tr.seed)

    # -- create timer --
    timer = ExpTimer()

    # -- paths --
    root = Path(cfgs.tr.root)
    log_dir = root / "output/train/logs/" / str(cfg.uuid)
    pik_dir = root / "output/train/pickles" / str(cfg.uuid)
    chkpt_dir = root / "output/train/checkpoints" / str(cfg.uuid)
    init_paths(log_dir,pik_dir,chkpt_dir)

    # -- network --
    model = ModelLit(cfs.net,cfgs.lit)
    # flow=cfg.flow,flow_method=cfg.flow_method,
    # isize=cfg.isize,batch_size=cfg.batch_size_tr,
    # lr_init=cfg.lr_init,lr_final=cfg.lr_final,
    # scheduler=cfg.scheduler,weight_decay=cfg.weight_decay,
    # nepochs=cfg.nepochs,task=cfg.task,
    # warmup_epochs=cfg.warmup_epochs,uuid=str(cfg.uuid),
    # sim_device=cfg.sim_device,sim_type=cfg.sim_type,
    # deno_clamp=cfg.deno_clamp,
    # optim=cfg.optim,momentum=cfg.momentum)

    # -- init validation performance --
    run_validation(log_dir,pik_dir,model,cfgs.tr)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #          Training
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    print(cfg.uuid)
    print("Num Training Vids: ",len(data.tr))
    print("Log Dir: ",log_dir)

    # -- pytorch_lightning training --
    logger = CSVLogger(log_dir,name="train",flush_logs_every_n_steps=1)
    ckpt_fn_val = cfg.uuid + "-{epoch:02d}-{val_loss:2.2e}"
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",save_top_k=10,
                                          mode="min",dirpath=chkpt_dir,
                                          filename=ckpt_fn_val)
    ckpt_fn_epoch = cfg.uuid + "-{epoch:02d}"
    cc_recent = ModelCheckpoint(monitor="epoch",save_top_k=10,mode="max",
                                dirpath=chkpt_dir,filename=ckpt_fn_epoch)
    callbacks = [checkpoint_callback,cc_recent]
    if cfg.swa:
        swa_callback = StochasticWeightAveraging(swa_lrs=cfg.lr_init,
                                                 swa_epoch_start=cfg.swa_epoch_start)
        callbacks += [swa_callback]
    trainer = pl.Trainer(accelerator="gpu",devices=cfg.ndevices,precision=32,
                         accumulate_grad_batches=cfg.accumulate_grad_batches,
                         limit_train_batches=cfg.limit_train_batches,
                         limit_val_batches=1.,max_epochs=cfg.nepochs,
                         log_every_n_steps=1,logger=logger,
                         gradient_clip_val=cfg.gradient_clip_val,
                         gradient_clip_algorithm=cfg.gradient_clip_algorithm,
                         callbacks=callbacks)
                         # strategy="ddp_find_unused_parameters_false")
    timer.start("train")

    # -- resume --
    ckpt_path = get_checkpoint(chkpt_dir,cfg.uuid,cfg.nepochs)
    trainer.fit(model, loaders.tr, loaders.val, ckpt_path=ckpt_path)
    timer.stop("train")
    best_model_path = checkpoint_callback.best_model_path


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #       Validation Testing
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- reload dataset with no isizes --
    # model.isize = None
    # cfg_clone = copy.deepcopy(cfg)
    # cfg_clone.isize = None
    # cfg_clone.cropmode = "center"
    cfg_clone.nsamples_tr = cfg.nsamples_at_testing
    cfg_clone.nsamples_val = cfg.nsamples_at_testing
    data,loaders = data_hub.sets.load(cfg_clone)

    # -- training performance --
    tr_report = MetricsCallback()
    logger = CSVLogger(log_dir,name="train_te",flush_logs_every_n_steps=1)
    trainer = pl.Trainer(gpus=1,precision=32,limit_train_batches=1.,
                         max_epochs=1,log_every_n_steps=1,
                         callbacks=[tr_report],logger=logger)
    timer.start("train_te")
    trainer.test(model, loaders.tr)
    timer.stop("train_te")
    tr_results = tr_report.metrics
    tr_res_fn = pik_dir / "train.pkl"
    write_pickle(tr_res_fn,tr_results)

    # -- validation performance --
    val_report = MetricsCallback()
    logger = CSVLogger(log_dir,name="val_te",flush_logs_every_n_steps=1)
    trainer = pl.Trainer(gpus=1,precision=32,limit_train_batches=1.,
                         max_epochs=1,log_every_n_steps=1,
                         callbacks=[val_report],logger=logger)
    timer.start("val_te")
    trainer.test(model, loaders.val)
    timer.stop("val_te")
    val_results = val_report.metrics
    print("--- Tuned Validation Results ---")
    print(val_results)
    val_res_fn = pik_dir / "val.pkl"
    write_pickle(val_res_fn,val_results)

    # -- report --
    results = edict()
    results.best_model_path = [best_model_path]
    results.init_val_results_fn = [init_val_res_fn]
    results.train_results_fn = [tr_res_fn]
    results.val_results_fn = [val_res_fn]
    results.train_time = [timer["train"]]
    results.test_train_time = [timer["train_te"]]
    results.test_val_time = [timer["val_te"]]
    results.test_init_val_time = [timer["init_val_te"]]
    for f,val in init_val_results.items():
        results["init_"+f] = val
    for f,val in val_results.items():
        results["final_"+f] = val
    print(results)

    return results

def init_paths(log_dir,pik_dir,chkpt_dir):

    # -- init log dir --
    print("Log Dir: ",log_dir)
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    log_subdirs = ["train"]
    for sub in log_subdirs:
        log_subdir = log_dir / sub
        if not log_subdir.exists(): log_subdir.mkdir()

    # -- prepare save directory for pickles --
    if not pik_dir.exists():
        pik_dir.mkdir(parents=True)

def get_checkpoint(checkpoint_dir,uuid,nepochs):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return ""
    prev_ckpt = ""
    for epoch in range(nepochs):
        ckpt_fn = checkpoint_dir / ("%s-epoch=%02d.ckpt" % (uuid,epoch))
        if ckpt_fn.exists(): prev_ckpt = ckpt_fn
        else: break
    assert ((prev_ckpt == "") or prev_ckpt.exists())
    if prev_ckpt != "":
        print("Resuming training from {%s}" % (str(prev_ckpt)))
    return str(prev_ckpt)


def run_validation(log_dir,pik_dir,model,cfg):

    # -- load dataset with testing mods isizes --
    cfg_clone = copy.deepcopy(cfg)
    cfg_clone.nsamples_val = cfg.nsamples_at_testing
    data,loaders = data_hub.sets.load(cfg_clone)

    # -- set model's isize --
    model.isize = cfg.isize

    # -- validation --
    init_val_report = MetricsCallback()
    logger = CSVLogger(log_dir,name="init_val_te",flush_logs_every_n_steps=1)
    trainer = pl.Trainer(accelerator="gpu",devices=1,precision=32,
                         limit_train_batches=1.,
                         max_epochs=3,log_every_n_steps=1,
                         callbacks=[init_val_report],logger=logger)
    timer.start("init_val_te")
    trainer.test(model, loaders.val)
    timer.stop("init_val_te")
    init_val_results = init_val_report.metrics
    print("--- Init Validation Results ---")
    print(init_val_results)
    init_val_res_fn = pik_dir / "init_val.pkl"
    write_pickle(init_val_res_fn,init_val_results)
    print(timer)

    # -- reset model --
    model.isize = cfg.isize


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def _accumulate_results(self,each_me):
        for key,val in each_me.items():
            if not(key in self.metrics):
                self.metrics[key] = []
            if hasattr(val,"ndim"):
                ndim = val.ndim
                val = val.cpu().numpy().item()
            self.metrics[key].append(val)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        print("logging metrics: ",metrics,step)

    def on_train_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_train_batch_end(self, trainer, pl_module, outs,
                           batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)


    def on_validation_batch_end(self, trainer, pl_module, outs,
                                batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_batch_end(self, trainer, pl_module, outs,
                          batch, batch_idx, dl_idx):
        self._accumulate_results(outs)
