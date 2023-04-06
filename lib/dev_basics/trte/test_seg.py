"""

Test segmentation methods.

"""

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
import tqdm
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- dev basics --
# from dev_basics.report import deno_report
from functools import partial
from dev_basics.aug_test import test_x8
from dev_basics import flow
from dev_basics import net_chunks
from dev_basics.utils.misc import get_region_gt
from dev_basics.utils.misc import optional,slice_flows,set_seed
from dev_basics.utils.misc import rslice,write_pickle,read_pickle
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.metrics import compute_psnrs,compute_ssims,compute_strred
from dev_basics.utils import vid_io

# -- local --
from .evals import get_evaluator

# -- config --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__)
extract_config = econfig.extract_config

def test_pairs():
    pairs = {"device":"cuda:0","seed":123,
             "frame_start":0,"frame_end":-1,
             "aug_test":False,"longest_space_chunk":False,
             "flow":False,"burn_in":False,"arch_name":None,
             "saved_dir":"./output/saved_examples/","uuid":"uuid_def",
             "flow_sigma":-1,"internal_adapt_nsteps":0,
             "internal_adapt_nepochs":0,"nframes":0,
             "save_deno":True,"python_module":"dev_basics.trte.id_model",
             "bench_bwd":False,"append_noise_map":False,"arch_name":"default",
             "test_task":"segm"}
    return pairs

@econfig.set_init
def run(cfg):

    # -- config --
    econfig.init(cfg)
    epairs = econfig.extract_pairs
    tcfg = epairs(cfg,test_pairs())
    module = econfig.required_module(tcfg,'python_module')
    model_cfg = epairs(module.extract_model_config(tcfg),cfg)
    if econfig.is_init: return
    if tcfg.frame_end == -1: tcfg.frame_end = tcfg.frame_start + cfg.nframes - 1

    # -- clear --
    th.cuda.empty_cache()
    th.cuda.synchronize()

    # -- set seed --
    set_seed(tcfg.seed)

    # -- set device --
    th.cuda.set_device(int(tcfg.device.split(":")[1]))

    # -- load eval --
    evaluator = get_evaluator(tcfg,tcfg.test_task)
    # evaluator.reset()

    # -- init results --
    # results = edict()
    results = edict({k:[] for k in evaluator.get_keys()})
    # results.psnrs = []
    # results.ssims = []
    # results.strred = []
    # results.noisy_psnrs = []
    # results.deno_fns = []
    results.vid_frames = []
    results.vid_name = []

    # -- init keyword fields --
    time_fields = ["flow","deno","attn","extract","search",
                   "agg","fold","fwd_grad","bwd"]
    for field in time_fields:
        results["timer_%s"%field] = []
    mem_fields = ["deno","adapt","fwd_grad","bwd"]
    for field in mem_fields:
        results["%s_mem_res"%field] = []
        results["%s_mem_alloc"%field] = []


    # -- burn_in once --
    burn_in = tcfg.burn_in

    # -- load model --
    model = module.load_model(model_cfg)

    # -- data --
    imax = 255.
    data,loaders = data_hub.sets.load(cfg)
    seg_tester = data.te.tester
    seg_eval = data.te.evals
    seg_eval.reset()
    num = len(seg_tester)

    # -- over the dataset --
    for index, inputs in tqdm.tqdm(enumerate(seg_tester),total=num):

        # -- create timer --
        timer = ExpTimer()
        memer = GpuMemer()

        # -- clean memory --
        th.cuda.empty_cache()
        # print("index: ",index)

        # -- unpack --
        # sample = data[cfg.dset][index]
        # region = sample['region']
        # noisy,clean = sample['noisy'][None,],sample['clean'][None,]
        # noisy,clean = noisy.to(tcfg.device),clean.to(tcfg.device)
        # vid_frames = sample['fnums'].numpy()
        B = len(inputs)
        image_ids = [inputs[b]['image_id'] for b in range(B)]
        clean = [inputs[b]['image'] for b in range(B)]
        clean = th.stack(clean)
        # print("[%d] clean.shape: " % index,clean.shape)

        # -- resample noise for flow --
        flow_img = clean
        if tcfg.flow_sigma >= 0:
            flow_img = th.normal(clean,tcfg.flow_sigma)

        # -- optical flow --
        with TimeIt(timer,"flow"):
            flows = flow.orun(flow_img,tcfg.flow,ftype="svnlb")

        # -- augmented testing --
        aug_fxn = model.forward
        fwd_fxn = model.forward

        # -- chunked processing --
        # chunk_cfg = net_chunks.extract_chunks_config(cfg)
        # if tcfg.longest_space_chunk:
        #     set_longest_spatial_chunk(chunk_cfg,noisy.shape)
        # fwd_fxn = net_chunks.chunk(chunk_cfg,aug_fxn)
        # chunk_fwd = fwd_fxn


        # -- denoise! --
        with MemIt(memer,"deno"):
            with TimeIt(timer,"deno"):
                with th.no_grad():
                    output = fwd_fxn(inputs,flows)
        mtimes = {}

        # -- save example --
        # res_info = evaluator.save_output(tcfg,output)
        # for k,v in res_info.items(): results[k] = v
        # out_dir = Path(tcfg.saved_dir) / tcfg.arch_name / str(tcfg.uuid)
        # if tcfg.save_deno:
        #     print("Saving Denoised Output [%s]" % out_dir)
        #     deno_fns = vid_io.save_video(deno,out_dir,"deno")
        # else:
        #     deno_fns = ["" for _ in range(deno.shape[0])]

        # -- deno quality metrics --
        # print(inputs)
        seg_eval.process(inputs,output)
        # seg_results = seg_eval.evaluate()
        # print(image_ids)
        # seg_res = seg_eval.evaluate(image_ids)
        # print(seg_res)
        # print(res_info)
        # for k,v in res_info.items(): results[k] = v
        # noisy_psnrs = compute_psnrs(noisy,clean,div=imax)
        # psnrs = compute_psnrs(clean,deno,div=imax)
        # ssims = compute_ssims(clean,deno,div=imax)
        # strred = compute_strred(clean,deno,div=imax)

        # -- compare [delete me] --
        # warps = model.warps
        # print(warps.shape)
        # ref = warps[:,:,:3]
        # div = ref.max().item()
        # K = 8
        # cmps = rearrange(warps[:,:,3:],'b t (k c) h w -> k b t c h w',k=K)
        # for k in range(K):
        #     # mse = th.mean((ref - cmps[k])**2)
        #     warp_psnrs = compute_psnrs(ref,cmps[k],div=div)
        #     print(k,warp_psnrs)
        # # exit(0)
        # print(psnrs,np.mean(psnrs))
        # import vrt
        # print(deno.shape,clean.shape)
        # psnrs = vrt.calculate_psnr(deno,clean)
        # print(psnrs,np.mean(psnrs),np.mean(ssims),np.mean(strred))

        # -- measure bwd info --
        if tcfg.bench_bwd:
            measure_bwd(model,fwd_fxn,flows,noisy/imax,
                        clean/imax,timer,memer)

        # -- append results --
        # results.psnrs.append(psnrs)
        # results.ssims.append(ssims)
        # results.strred.append(strred)
        # results.noisy_psnrs.append(noisy_psnrs)
        # results.deno_fns.append(deno_fns)
        # results.vid_frames.append(vid_frames)
        # results.vid_name.append([cfg.vid_name])
        for name,(mem_res,mem_alloc) in memer.items():
            key = "%s_%s" % (name,"mem_res")
            results[key].append([mem_res])
            key = "%s_%s" % (name,"mem_alloc")
            results[key].append([mem_alloc])
        for name,time in timer.items():
            if not(name in results):
                results[name] = []
            results[name].append(time)
        for name,time in mtimes.items():
            if not(name in results):
                results[name] = []
            results[name].append(time)

    # -- clear --
    th.cuda.empty_cache()
    th.cuda.synchronize()

    # -- results --
    seg_results = seg_eval.evaluate()
    print(seg_results)

    return results


def measure_bwd(model,fwd_fxn,flows,noisy,clean,timer,memer):

    # -- train mode --
    model.train()

    # -- forward pass again --
    with MemIt(memer,"fwd_grad"):
        with TimeIt(timer,"fwd_grad"):
            deno = fwd_fxn(noisy,flows)
    if hasattr(model,'reset_times'):
        model.reset_times()

    # -- backward pass! --
    with MemIt(memer,"bwd"):
        with TimeIt(timer,"bwd"):
            loss = th.mean((deno - clean)**2)
            loss.backward()

    # -- test mode again --
    model.eval()
