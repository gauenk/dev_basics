
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
from data_hub.common import get_loaders,optional,get_isize

# -- modules --
# import importlib

# -- pytorch-lit --
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging

# -- ddp --
import time
# from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_only

# -- bench --
from . import bench

# -- wandb --
WANDB_AVAIL = False
try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    WANDB_AVAIL = True
except:
    pass

# -- dev basics --
from .. import flow
from ..utils.misc import set_seed,optional
from ..utils.misc import write_pickle
from ..utils.timer import ExpTimer,TimeIt
from ..utils.metrics import compute_psnrs,compute_ssims

# -- extract config --
from ..configs import ExtractConfig,dcat
econfig = ExtractConfig(__file__)
extract_config = econfig.extract_config

def train_pairs():
    pairs = {"num_workers":4,
             "dset_tr":"tr",
             "dset_val":"val",
             "persistent_workers":True,
             "rand_order_tr":True,
             "gradient_clip_algorithm":"norm",
             "gradient_clip_val":0.0,
             "index_skip_val":5,
             "root":".","seed":123,
             "accumulate_grad_batches":1,
             "ndevices":1,
             "num_nodes":1,
             "precision":32,
             "limit_train_batches":1.,
             "nepochs":30,
             "offset_seed_rank":False,
             "uuid":"",
             "swa":False,
             "swa_epoch_start":0.8,
             "nsamples_at_testing":1,
             "isize":"128_128",
             "subdir":"",
             "save_epoch_list":"",
             "use_wandb":False
    }
    return pairs

def load_cls_data(cfg,name,root,nworkers):

    # -- get data --
    data = edict()
    data.tr = torchvision.datasets.CIFAR10(root=root, train=True,
                                           download=True, transform=transform)
    data.val = torchvision.datasets.CIFAR10(root=root, train=False,
                                            download=True, transform=transform)
    data.te = data.val

    # -- get loaders --
    dcfg = edict()
    dcfg.num_workers = nworkers
    modes = ['tr','val','te']
    fields = {"batch_size":1}
    p = parse_cfg(cfg,modes,fields)
    batch_size = edict({key:val['batch_size'] for key,val in p.items()})
    loader = get_loaders(dcfg,data,batch_size)

    return data,loaders

def overwrite_nepochs(cfg,nepochs):
    if nepochs is None: return
    print("Manually overwriting nepoch from %d to %d" % (cfg.nepochs,nepochs))
    cfg.nepochs = nepochs

def overwrite_field(name,cfg,value):
    if value is None: return
    print("Manually overwriting %s from %s to %s" % (name,str(cfg[name]),str(value)))
    cfg[name] = value

@econfig.set_init
def run(cfg,nepochs=None,flow_from_end=None,flow_epoch=None):

    # -=-=-=-=-=-=-=-=-
    #
    #     Init Exp
    #
    # -=-=-=-=-=-=-=-=-

    # -- config --
    econfig.init(cfg)
    net_module = econfig.required_module(cfg,"python_module")
    lit_module = net_module.lightning
    sim_module = econfig.optional_module(cfg,"sim_module")
    net_extract_config = net_module.extract_config
    lit_extract_config = lit_module.extract_config
    sim_extract_config = sim_module.extract_config
    cfgs = econfig.extract_set({"tr":train_pairs(),
                                "net":net_extract_config(cfg),
                                "lit":lit_extract_config(cfg),
                                "sim":sim_extract_config(cfg)})
    cfg = dcat(cfg,econfig.flatten(cfgs)) # update cfg
    if econfig.is_init: return
    # overwrite_nepochs(cfgs.tr,nepochs)
    overwrite_field("nepochs",cfgs.tr,nepochs)
    overwrite_field("nepochs",cfgs.lit,nepochs)
    overwrite_field("flow_from_end",cfgs.lit,flow_from_end)
    overwrite_field("flow_epoch",cfgs.lit,flow_epoch)
    if cfgs.tr.gradient_clip_val <= 0:
        cfgs.tr.gradient_clip_val = None

    # -- setup process --
    print("PID: ",os.getpid())
    th.set_float32_matmul_precision('medium')

    # -- set seed --
    seed_num = cfgs.tr.seed
    if cfg.offset_seed_rank:
        seed_num += rank_zero_only.rank
    set_seed(seed_num)

    # -- init model/simulator/lightning --
    device = "cuda"
    print(th.cuda.current_device())
    # th.cuda.device(device)
    net = net_module.load_model(cfgs.net).to(device)
    sim = getattr(sim_module,cfgs.sim.load_fxn)(cfgs.sim)
    model = lit_module.LitModel(cfgs.lit,net,sim)

    # -- init torch --
    th.set_float32_matmul_precision('medium')

    # -- net summary --
    # vshape = (4,4,3,256,256)
    # res = bench.summary_loaded(net,vshape,with_flows=True)
    # print("-="*25+"-")
    # print("Network Summary for %s" % str(vshape))
    # print(res)
    # print("-="*25+"-")

    # -- set-up --
    print("PID: ",os.getpid())
    seed_num = cfgs.tr.seed
    if cfg.offset_seed_rank:
        seed_num += rank_zero_only.rank
    set_seed(seed_num)
    cfgs.tr.use_wandb = cfgs.tr.use_wandb

    # -- create timer --
    timer = ExpTimer()

    # -- paths --
    if cfgs.tr.subdir != "":
        name = cfgs.tr.subdir
    name = cfgs.tr.name
    root = Path(cfgs.tr.root) / "output" / "train" / name
    log_dir = root / "logs" / str(cfgs.tr.uuid)
    pik_dir = root / "pickles" / str(cfgs.tr.uuid)
    chkpt_dir = root / "checkpoints" / str(cfgs.tr.uuid)
    if rank_zero_only.rank == 0:
        init_paths(log_dir,pik_dir,chkpt_dir)
    else:
        time.sleep(5)

    # -- copy previous step checkpoint --
    # input: previous step's uuid

    # -- init validation performance --
    # outs = run_validation(cfg,log_dir,pik_dir,timer,model,"val","init_val_te")
    # init_val_results,init_val_res_fn = outs
    # init_val_results,init_val_res_fn = {"init_val_te":-1},""

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #          Training
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- data --
    # dset_tr = cfgs.tr.dset_tr
    # dset_val = cfgs.tr.dset_val
    name = "cifar10"
    root = "~/Documents/data/"
    nworkers = 2
    data,loaders = load_cls_data(cfg,name,root,nworkers)
    print("Num Training Vids: ",len(data[dset_tr]))
    print("Log Dir: ",log_dir)

    # -- pytorch_lightning training --
    trainer,chkpt_callback = create_trainer(cfgs,log_dir,chkpt_dir)
    ckpt_path = get_checkpoint(chkpt_dir,cfgs.tr.uuid,cfgs.tr.nepochs)
    print(len(loaders[dset_val]),type(loaders[dset_val]))
    print("Checkpoint Path: %s" % str(ckpt_path))
    # exit()
    timer.start("train")
    trainer.fit(model, loaders[dset_tr], loaders[dset_val], ckpt_path=ckpt_path)
    timer.stop("train")
    best_model_path = chkpt_callback.best_model_path


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #       Validation Testing
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- training performance --
    outs = run_validation(cfg,log_dir,pik_dir,timer,model,"tr","train_te")
    tr_results,tr_res_fn = outs

    # -- validation performance --
    outs = run_validation(cfg,log_dir,pik_dir,timer,model,"val","val_te")
    val_results,val_res_fn = outs

    # -- report --
    results = edict()
    results.best_model_path = [best_model_path]
    results.init_val_results_fn = [init_val_res_fn]
    results.train_results_fn = [tr_res_fn]
    results.val_results_fn = [val_res_fn]
    results.train_time = [timer["train"]]
    results.test_init_val_time = [timer["init_val_te"]]
    results.test_train_time = [timer["train_te"]]
    results.test_val_time = [timer["val_te"]]
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
        if not log_subdir.exists():
            log_subdir.mkdir(parents=True)

    # -- prepare save directory for pickles --
    if not pik_dir.exists():
        pik_dir.mkdir(parents=True)

def wait_checkpoint_exists(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    while not(ckpt_dir.exists()):
        time.sleep(1)

def get_checkpoint(checkpoint_dir,uuid,nepochs):
    checkpoint_dir = Path(checkpoint_dir)
    if rank_zero_only.rank > 0:
        wait_checkpoint_exists(checkpoint_dir)
    else:
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True)
            return None
    chosen_ckpt = ""
    for epoch in range(nepochs):
        # if epoch > 49: break
        ckpt_fn = checkpoint_dir / ("%s-epoch=%02d.ckpt" % (uuid,epoch))
        if ckpt_fn.exists(): chosen_ckpt = ckpt_fn
    assert ((chosen_ckpt == "") or chosen_ckpt.exists())
    if chosen_ckpt != "":
        print("Resuming training from {%s}" % (str(chosen_ckpt)))
        chosen_ckpt = str(chosen_ckpt)
    else:
        chosen_ckpt = None
    return chosen_ckpt

def create_trainer(cfgs,log_dir,chkpt_dir):
    logger = get_logger(log_dir,"train",cfgs.tr.use_wandb)
    ckpt_fn_val = cfgs.tr.uuid + "-{epoch:02d}-{val_loss:2.2e}"
    checkpoint_list = SaveCheckpointList(cfgs.tr.uuid,chkpt_dir,cfgs.tr.save_epoch_list)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",save_top_k=3,
                                          mode="min",dirpath=chkpt_dir,
                                          filename=ckpt_fn_val)
    ckpt_fn_epoch = cfgs.tr.uuid + "-{epoch:02d}"
    cc_recent = ModelCheckpoint(monitor="epoch",save_top_k=3,mode="max",
                                dirpath=chkpt_dir,filename=ckpt_fn_epoch)
    callbacks = [checkpoint_list,checkpoint_callback,cc_recent]
    if cfgs.tr.swa:
        swa_callback = StochasticWeightAveraging(swa_lrs=cfgs.tr.lr_init,
                                swa_epoch_start=cfgs.tr.swa_epoch_start)
        callbacks += [swa_callback]
    ndevices_local = int(cfgs.tr.ndevices/cfgs.tr.num_nodes)
    print(cfgs.tr.num_nodes,cfgs.tr.ndevices,ndevices_local)
    trainer = pl.Trainer(accelerator="gpu",
                         # num_nodes=cfgs.tr.num_nodes,
                         devices=ndevices_local,precision=32,
                         accumulate_grad_batches=cfgs.tr.accumulate_grad_batches,
                         limit_train_batches=cfgs.tr.limit_train_batches,
                         limit_val_batches=1.,max_epochs=cfgs.tr.nepochs,
                         log_every_n_steps=1,logger=logger,
                         gradient_clip_val=cfgs.tr.gradient_clip_val,
                         gradient_clip_algorithm=cfgs.tr.gradient_clip_algorithm,
                         callbacks=callbacks,detect_anomaly=True,
                         strategy="ddp_find_unused_parameters_false")

    return trainer,checkpoint_callback

def get_logger(log_dir,name,use_wandb):
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
    GLOBAL_RANK = int(os.environ.get('GLOBAL_RANK', 0))
    NODE_RANK_0 = (LOCAL_RANK == 0) and (GLOBAL_RANK == 0)
    if use_wandb and WANDB_AVAIL and rank_zero_only.rank == 0:
        logger = WandbLogger(name=name)
    elif use_wandb and not( rank_zero_only.rank == 0):
        logger = None
    else:
        logger = CSVLogger(log_dir,name=name,flush_logs_every_n_steps=1)
    print(logger)

    return logger

def run_validation(cfg,log_dir,pik_dir,timer,model,dset,name):

    # -- load dataset with testing mods isizes --
    cfg_clone = copy.deepcopy(cfg)
    cfg_clone.dname = optional(cfg,"dname_at_testing",cfg.dname)
    cfg_clone.nframes = optional(cfg,"nframes_at_testing",5)
    cfg_clone.nsamples_tr = optional(cfg,"nsamples_te_at_testing",0)
    cfg_clone.nsamples_val = optional(cfg,"nsamples_at_testing",0)
    cfg_clone.nsamples_te = optional(cfg,"nsamples_at_testing",0)
    cfg_clone.isize = optional(cfg,"isize_at_testing",None)
    cfg_clone.batch_size = 1
    cfg_clone.batch_size_tr = 1
    cfg_clone.batch_size_val = 1
    cfg_clone.batch_size_te = 1
    data,loaders = data_hub.sets.load(cfg_clone)
    print("len(data.val): ",len(data.val))
    print("len(loaders.val): ",len(loaders.val))

    # -- set model's isize --
    model.isize = cfg.isize

    # -- setup --
    val_report = MetricsCallback()
    logger = get_logger(log_dir,"val",cfg.use_wandb)
    ndevices_local = int(cfg.ndevices/cfg.num_nodes)
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,#ndevices_local,
                         precision=32,
                         num_nodes=1,#cfg.num_nodes,
                         limit_train_batches=1.,
                         max_epochs=3,log_every_n_steps=1,
                         callbacks=[val_report],logger=logger)

    # -- run --
    timer.sync_start(name)
    trainer.test(model, loaders[dset])
    timer.sync_stop(name)

    # -- unpack results --
    results = val_report.metrics
    print("--- Results [%s] ---" % name)
    print(results)
    res_fn = pik_dir / ("%s.pkl" % name)
    write_pickle(res_fn,results)
    print(timer)

    # -- reset model --
    model.isize = cfg.isize
    return results,res_fn

class SaveCheckpointList(Callback):

    def __init__(self,uuid,outdir,save_epochs):
        super().__init__()
        self.uuid = uuid
        self.outdir = outdir
        self.save_epochs = []
        self.save_interval = -1
        if "-" in save_epochs:
            self.save_epochs = [int(s) for s in save_epochs.split("-")]
            self.save_type = "list"
        elif save_epochs.startswith("by"):
            self.save_interval = int(save_epochs.split("by")[-1])
            self.save_type = "interval"

    def on_train_epoch_end(self, trainer, pl_module):
        uuid = self.uuid
        epoch = trainer.current_epoch
        if self.save_type == "list":
            if not(epoch in self.save_epochs): return
            path = Path(self.outdir / ("%s-save-epoch=%02d.ckpt" % (uuid,epoch)))
            trainer.save_checkpoint(str(path))
        elif self.save_type == "interval":
            if not((epoch+1) % self.save_interval == 0): return
            path = Path(self.outdir / ("%s-save-epoch=%02d.ckpt" % (uuid,epoch)))
            trainer.save_checkpoint(str(path))

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def _accumulate_results(self,each_me):
        for key,val in each_me.items():
            # print(key,val)
            if not(key in self.metrics):
                self.metrics[key] = []
            if hasattr(val,"ndim"):
                ndim = val.ndim
                if isinstance(val,th.tensor):
                    val = val.cpu().numpy().item()
                else:
                    val = val
            self.metrics[key].append(val)

    # @rank_zero_only
    # def log_metrics(self, metrics, step):
    #     # metrics is a dictionary of metric names and values
    #     # your code to record metrics goes here
    #     print("logging metrics: ",metrics,step)

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
                           batch, batch_idx, dataloader_idx=0):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_validation_batch_end(self, trainer, pl_module, outs,
                                batch, batch_idx, dataloader_idx=0):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_batch_end(self, trainer, pl_module, outs,
                          batch, batch_idx, dataloader_idx=0):
        self._accumulate_results(outs)

