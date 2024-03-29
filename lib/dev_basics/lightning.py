

# -- misc --
import os,math,tqdm,sys
import pprint,copy
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- optical flow --
from dev_basics import flow

# -- caching results --
import cache_io

# # -- network --
# import nlnet

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__)
extract_config = econfig.extract_config

# -- misc --
from dev_basics.utils.misc import rslice,write_pickle,read_pickle
from dev_basics.utils.metrics import compute_psnrs,compute_ssims
from dev_basics.utils.timer import ExpTimer
import dev_basics.utils.gpu_mem as gpu_mem

# -- noise sims --
import importlib
# try:
#     import stardeno
# except:
#     pass

# # -- wandb --
# WANDB_AVAIL = False
# try:
#     import wandb
#     WANDB_AVAIL = True
# except:
#     pass

# -- generic logging --
import logging
logging.basicConfig()

# -- lightning module --
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# import torch
# torch.autograd.set_detect_anomaly(True)

@econfig.set_init
def init_cfg(cfg):
    econfig.init(cfg)
    cfgs = econfig.extract_dict_of_pairs(cfg,{"lit":lit_pairs(),
                                              "sim":sim_pairs()},
                                         restrict=True)
    return cfgs

def lit_pairs():
    pairs = {"batch_size":1,"flow":True,"flow_method":"cv2",
             "isize":None,"bw":False,"lr_init":1e-3,
             "lr_final":1e-8,"weight_decay":0.,
             "nepochs":0,"task":"denoising","uuid":"",
             "scheduler_name":"default","step_lr_size":5,
             "step_lr_gamma":0.1,"flow_epoch":None,
             "flow_from_end":None,"use_wandb":False,
             "ntype":"g","rate":-1,"sigma":-1,
             "sigma_min":-1,"sigma_max":-1,
             "optim_name":"adam",
             "sgd_momentum":0.1,"sgd_dampening":0.1,
             "coswr_T0":-1,"coswr_Tmult":1,"coswr_eta_min":1e-9,
             "step_lr_multisteps":"30-50"}
    return pairs

def sim_pairs():
    pairs = {"sim_type":"g","sim_module":"stardeno",
             "sim_device":"cuda:0","load_fxn":"load_sim"}
    return pairs

def get_sim_model(self,cfg):
    if cfg.sim_type == "g":
        return None
    elif cfg.sim_type == "stardeno":
        module = importlib.load_module(cfg.sim_module)
        return module.load_noise_sim(cfg.sim_device,True).to(cfg.sim_device)
    else:
        raise ValueError(f"Unknown sim model [{sim_type}]")

class LitModel(pl.LightningModule):

    def __init__(self,lit_cfg,net,sim_model):
        super().__init__()
        lit_cfg = init_cfg(lit_cfg).lit
        for key,val in lit_cfg.items():
            setattr(self,key,val)
        self.set_flow_epoch() # only for current exps; makes last 10 epochs with opt. flow.
        self.net = net
        self.sim_model = sim_model
        self.gen_loger = logging.getLogger('lightning')
        self.gen_loger.setLevel("NOTSET")
        self.automatic_optimization=True

    def forward(self,vid,flows=None):
        if flows is None:
            flows = flow.orun(vid,self.flow,ftype=self.flow_method)
        deno = self.net(vid,flows=flows)
        return deno

    def sample_noisy(self,batch):
        if self.sim_model is None: return
        clean = batch['clean']
        noisy = self.sim_model.run_rgb(clean)
        batch['noisy'] = noisy

    def set_flow_epoch(self):
        if not(self.flow_epoch is None): return
        if self.flow_from_end is None: return
        if self.flow_from_end == 0: return
        self.flow_epoch = self.nepochs - self.flow_from_end

    def update_flow(self):
        if self.flow_epoch is None: return
        if self.flow_epoch <= 0: return
        if self.current_epoch >= self.flow_epoch:
            self.flow = True

    def configure_optimizers(self):
        if self.optim_name == "adam":
            optim = th.optim.Adam(self.parameters(),lr=self.lr_init,
                                  weight_decay=self.weight_decay)
        elif self.optim_name == "adamw":
            optim = th.optim.AdamW(self.parameters(),lr=self.lr_init,
                                   weight_decay=self.weight_decay)
        elif self.optim_name == "sgd":
            optim = th.optim.SGD(self.parameters(),lr=self.lr_init,
                                 weight_decay=self.weight_decay,
                                 momentum=self.sgd_momentum,
                                 dampening=self.sgd_dampening)
        else:
            raise ValueError(f"Unknown optim [{self.optim_name}]")
        sched = self.configure_scheduler(optim)
        return [optim], [sched]

    def configure_scheduler(self,optim):
        print("self.scheduler_name: ",self.scheduler_name)
        if self.scheduler_name in ["default","exp_decay"]:
            gamma = math.exp(math.log(self.lr_final/self.lr_init)/self.nepochs)
            ExponentialLR = th.optim.lr_scheduler.ExponentialLR
            scheduler = ExponentialLR(optim,gamma=gamma) # (.995)^50 ~= .78
            scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        elif self.scheduler_name in ["step","steplr"]:
            args = (self.step_lr_size,self.step_lr_gamma)
            # print("[Scheduler]: StepLR(%d,%2.2f)" % args)
            StepLR = th.optim.lr_scheduler.StepLR
            scheduler = StepLR(optim,step_size=self.step_lr_size,
                               gamma=self.step_lr_gamma)
        elif self.scheduler_name in ["cosa"]:
            CosAnnLR = th.optim.lr_scheduler.CosineAnnealingLR
            scheduler = CosAnnLR(optim,self.nepochs)
            scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        elif self.scheduler_name in ["cosa_step"]:
            print("cosa_step: ",self.scheduler_name)
            print("num_steps: ",num_steps)
            nsteps = self.num_steps()
            CosAnnLR = th.optim.lr_scheduler.CosineAnnealingLR
            scheduler = CosAnnLR(optim,nsteps)
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif self.scheduler_name in ["multi_step"]:
            milestones = [int(x) for x in self.step_lr_multisteps.split("-")]
            MultiStepLR = th.optim.lr_scheduler.MultiStepLR
            scheduler = MultiStepLR(optim,milestones=milestones,
                                    gamma=self.step_lr_gamma)
            scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        elif self.scheduler_name in ["coswr","cosw"]:
            lr_sched =th.optim.lr_scheduler
            CosineAnnealingWarmRestarts = lr_sched.CosineAnnealingWarmRestarts
            # print(self.coswr_T0,self.coswr_Tmult,self.coswr_eta_min)
            scheduler = CosineAnnealingWarmRestarts(optim,self.coswr_T0,
                                                    T_mult=self.coswr_Tmult,
                                                    eta_min=self.coswr_eta_min)
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif self.scheduler_name in ["none"]:
            StepLR = th.optim.lr_scheduler.StepLR
            scheduler = StepLR(optim,step_size=10**3,gamma=1.)
        else:
            raise ValueError(f"Uknown scheduler [{self.scheduler_name}]")
        return scheduler

    def training_step(self, batch, batch_idx):

        # -- sample noise from simulator --
        self.sample_noisy(batch)

        # -- update flow --
        self.update_flow()

        # -- each sample in batch --
        loss = 0 # init @ zero
        denos,cleans = [],[]
        ntotal = len(batch['noisy'])
        nbatch = ntotal
        nbatches = (ntotal-1)//nbatch+1
        for i in range(nbatches):
            start,stop = i*nbatch,min((i+1)*nbatch,ntotal)
            deno_i,clean_i,loss_i = self.training_step_i(batch, start, stop)
            loss += loss_i
            denos.append(deno_i)
            cleans.append(clean_i)
        loss = loss / nbatches

        # -- view params --
        # loss.backward()
        # for name, param in self.net.named_parameters():
        #     if param.grad is None:
        #         print(name)

        # -- append --
        denos = th.cat(denos)
        cleans = th.cat(cleans)

        # -- log --
        val_psnr = np.mean(compute_psnrs(denos,cleans,div=1.)).item()
        # val_ssim = np.mean(compute_ssims(denos,cleans,div=1.)).item() # too slow.
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=False, batch_size=self.batch_size)
        self.log("train_psnr", val_psnr, on_step=True,
                 on_epoch=False, batch_size=self.batch_size)
        lr = self.optimizers()._optimizer.param_groups[-1]['lr']
        self.log("lr", lr, on_step=True,
                 on_epoch=False, batch_size=self.batch_size)
        self.log("global_step", self.global_step, on_step=True,
                 on_epoch=False, batch_size=self.batch_size)
        # self.log("train_ssim", val_ssim, on_step=True,
        #          on_epoch=False, batch_size=self.batch_size)
        self.gen_loger.info("train_psnr: %2.2f" % val_psnr)

        return loss

    def training_step_i(self, batch, start, stop):

        # -- unpack batch
        noisy = batch['noisy'][start:stop]/255.
        clean = batch['clean'][start:stop]/255.
        fflow = batch['fflow'][start:stop]
        bflow = batch['bflow'][start:stop]

        # -- make flow --
        if fflow.shape[-2:] == noisy.shape[-2:]:
            flows = edict({"fflow":fflow,"bflow":bflow})
        else:
            flows = None

        # -- foward --
        deno = self.forward(noisy,flows)

        # -- report loss --
        loss = th.mean((clean - deno)**2)
        return deno.detach(),clean,loss

    def validation_step(self, batch, batch_idx):

        # -- sample noise from simulator --
        self.sample_noisy(batch)

        # -- denoise --
        noisy,clean = batch['noisy']/255.,batch['clean']/255.
        val_index = batch['index'].cpu().item()

        # -- flow --
        fflow = batch['fflow']
        bflow = batch['bflow']
        if fflow.shape[-2:] == noisy.shape[-2:]:
            flows = edict({"fflow":fflow,"bflow":bflow})
        else:
            flows = None

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with th.no_grad():
            deno = self.forward(noisy,flows)
        mem_res,mem_alloc = gpu_mem.print_peak_gpu_stats(False,"val",reset=True)

        # -- loss --
        loss = th.mean((clean - deno)**2)
        val_psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        val_ssim = np.mean(compute_ssims(deno,clean,div=1.)).item()

        # -- report --
        self.log("val_loss", loss.item(), on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_mem_res", mem_res, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_mem_alloc", mem_alloc, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_psnr", val_psnr, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_ssim", val_ssim, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_index", val_index, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("global_step",self.global_step,on_step=False,
                 on_epoch=True,batch_size=1)
        self.gen_loger.info("val_psnr: %2.2f" % val_psnr)
        self.gen_loger.info("val_ssim: %.3f" % val_ssim)


    def test_step(self, batch, batch_nb):

        # -- sample noise from simulator --
        self.sample_noisy(batch)

        # -- denoise --
        index = float(batch['index'][0].item())
        noisy,clean = batch['noisy']/255.,batch['clean']/255.

        # -- flow --
        fflow = batch['fflow']
        bflow = batch['bflow']
        if fflow.shape[-2:] == noisy.shape[-2:]:
            flows = edict({"fflow":fflow,"bflow":bflow})
        else:
            flows = None

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"test",reset=True)
        with th.no_grad():
            deno = self.forward(noisy,flows)
        mem_res,mem_alloc = gpu_mem.print_peak_gpu_stats(False,"test",reset=True)

        # -- compare --
        loss = th.mean((clean - deno)**2)
        psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        ssim = np.mean(compute_ssims(deno,clean,div=1.)).item()

        # -- terminal log --
        self.log("test_psnr", psnr, on_step=True, on_epoch=False, batch_size=1)
        self.log("test_ssim", ssim, on_step=True, on_epoch=False, batch_size=1)
        self.log("test_index", index,on_step=True,on_epoch=False,batch_size=1)
        self.log("test_mem_res", mem_res, on_step=True, on_epoch=False, batch_size=1)
        self.log("test_mem_alloc", mem_alloc,on_step=True,on_epoch=False,batch_size=1)
        self.log("global_step",self.global_step,on_step=True,on_epoch=False,batch_size=1)
        self.gen_loger.info("te_psnr: %2.2f" % psnr)
        self.gen_loger.info("te_ssim: %.3f" % ssim)

        # -- log --
        results = edict()
        results.test_loss = loss.item()
        results.test_psnr = psnr
        results.test_ssim = ssim
        results.test_mem_alloc = mem_alloc
        results.test_mem_res = mem_res
        results.test_index = index#.cpu().numpy().item()
        return results

    def num_steps(self) -> int:
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        dataset = self.trainer.fit_loop._data_source.dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, self.trainer.num_devices)
        acc = self.trainer.accumulate_grad_batches
        num_steps = dataset_size * self.trainer.max_epochs // (acc * num_devices)
        return num_steps

def remove_lightning_load_state(state):
    names = list(state.keys())
    for name in names:
        name_new = name.split(".")[1:]
        name_new = ".".join(name_new)
        state[name_new] = state[name]
        del state[name]
