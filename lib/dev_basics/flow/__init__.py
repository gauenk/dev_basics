"""
Wrap the opencv optical flow

"""

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- estimate sigma --
from skimage.restoration import estimate_sigma

# -- misc --
from easydict import EasyDict as edict

# -- opencv --
import cv2 as cv

# -- svnb --
try:
    import svnlb
except:
    pass

# -- rescale --
import torch.nn.functional as tnnf

# -- local --
from ..utils import color
from ..utils.misc import rslice as rslice_tensor

def run_zeros(vid,sigma=0.):
    device = vid.device
    b,t,c,h,w = vid.shape
    flows = edict()
    flows.fflow = th.zeros((b,t,2,h,w),device=device)
    flows.bflow = th.zeros((b,t,2,h,w),device=device)
    return flows

def orun(noisy,run_bool=True,sigma=None,ftype="cv2"): # optional run
    if run_bool:
        if sigma is None:
            sigma_est = est_sigma(noisy)
        else:
            sigma_est = sigma
        if len(noisy.shape) == 4:
            noisy = noisy[None,:]
        flows = run_batch(noisy,sigma_est,ftype)
    else:
        if len(noisy.shape) == 4:
            noisy = noisy[None,:]
        flows = run_zeros(noisy)

    return flows


def run_batch(vid,sigma=None,ftype="cv2"):
    sigma = get_sigma(vid,sigma)
    B = vid.shape[0]
    flows = edict()
    flows.fflow,flows.bflow = [],[]
    for b in range(B):
        flows_b = run(vid[b],sigma,ftype)
        flows.fflow.append(flows_b.fflow)
        flows.bflow.append(flows_b.bflow)
    flows.fflow = th.stack(flows.fflow)
    flows.bflow = th.stack(flows.bflow)
    return flows

def run(vid_in,sigma=None,ftype="cv2"):
    sigma = get_sigma(vid_in,sigma)
    if "cv2" in ftype:
        fxn_s,dev_s = get_fxn_s(ftype)
        return run_cv2(vid_in,sigma,fxn_s,dev_s)
    elif ftype == "svnlb":
        return run_svnlb(vid_in,sigma)
    else:
        raise ValueError(f"Uknown flow type [{ftype}]")

def get_sigma(vid,sigma):
    if sigma is None:
        sigma_est = est_sigma(vid)
    else:
        sigma_est = sigma
    return sigma_est

def get_fxn_s(ftype):
    # cv2_<function_string_here>
    if not("_" in ftype):
        return "farne","cpu"
    else:
        name = ftype.split("_")[1:]
        if len(name) == 1:
            return name[0],"gpu"
        elif len(name) == 2:
            return name[0],name[1]
        else:
            raise ValueError("Should only by cv2_<fname>_<cpu/gpu>")
#
# -- details --
#

def run_svnlb(vid_in,sigma,rescale=True):

    # -- rescale --
    if rescale:
        vid_in = vid_in*255.

    # -- run --
    vid_in_c = vid_in.cpu().numpy()
    if vid_in_c.shape[-3] == 1:
        vid_in_c = repeat(vid_in_c,'b 1 h w -> b c h w',c=3)
    fflow,bflow = svnlb.swig.runPyFlow(vid_in_c,sigma)

    # -- packing --
    flows = edict()
    flows.fflow = th.from_numpy(fflow).to(vid_in.device)
    flows.bflow = th.from_numpy(bflow).to(vid_in.device)

    # -- append zeros --
    zflow = th.zeros_like(flows.fflow[[0]])
    flows.fflow = th.cat([flows.fflow,zflow],0)
    flows.bflow = th.cat([zflow,flows.bflow],0)

    return flows

def run_cv2(vid_in,sigma,fxn_s,dev_s,rescale=True):

    # -- get flow function --
    flow_fxn = get_flow_fxn(fxn_s,dev_s)

    # -- init --
    device = vid_in.device
    vid_in = vid_in.cpu()
    vid = vid_in.clone() # copy data for no-rounding-error from RGB <-> YUV
    t,c,h,w = vid.shape

    # -- rescale --
    if rescale:
        vid = vid*255.

    # -- color2gray --
    vid = th.clamp(vid,0,255.).type(th.uint8)
    if vid.shape[1] == 3:
        color.rgb2yuv(vid)
    vid = vid[:,[0],:,:]
    vid = rearrange(vid,'t c h w -> t h w c')

    # -- alloc --
    fflow = th.zeros((t,2,h,w),device=device)
    bflow = th.zeros((t,2,h,w),device=device)

    # -- computing --
    for ti in reversed(range(t-1)):
        bflow[ti+1] = pair2flow(vid[ti+1],vid[ti],flow_fxn,device)
    for ti in range(t-1):
        fflow[ti] = pair2flow(vid[ti],vid[ti+1],flow_fxn,device)

    # -- final shaping --
    # fflow = rearrange(fflow,'t h w c -> t c h w')
    # bflow = rearrange(bflow,'t h w c -> t c h w')

    # -- packing --
    flows = edict()
    flows.fflow = fflow
    flows.bflow = bflow

    # -- gray2color --
    color.yuv2rgb(vid)

    return flows

def est_sigma(vid):
    if vid.ndim == 5:
        vid = vid[0]
    if vid.shape[1] == 3:
        vid = vid.cpu().clone()
        color.rgb2yuv(vid)
    vid_np = vid.cpu().numpy()
    vid_np = vid_np[:,[0]] # Y only
    sigma = estimate_sigma(vid_np,channel_axis=1)[0]
    return sigma

def get_flow_fxn(fxn_s,dev_s):
    # print("dev_s: ",dev_s)
    if dev_s == "gpu":
        return get_flow_fxn_gpu(fxn_s)
    elif dev_s == "cpu":
        return get_flow_fxn_cpu(fxn_s)
    else:
        raise ValueError("Uknown device [%s]" % dev_s)

def get_flow_fxn_gpu(fxn_s):
    if fxn_s == "farne":
        def wrapper(frame_curr,frame_next):
            frame_curr,frame_next = pair2gpu(frame_curr,frame_next)
            optical_flow = cv.cuda.FarnebackOpticalFlow_create()
            flow = optical_flow.calc(frame_curr,frame_next,None)
            return flow.download()
    elif fxn_s == "tvl1":
        def wrapper(frame_curr,frame_next):
            frame_curr,frame_next = pair2gpu(frame_curr,frame_next)
            optical_flow = cv.cuda.OpticalFlowDual_TVL1_create()
            flow = optical_flow.calc(frame_curr, frame_next, None)
            return flow.download()
    elif fxn_s == "tvl1p":
        def wrapper(frame_curr,frame_next):
            args = {"Tau":0.25,"Lambda":0.15,"Theta":0.3,"NumScales":5,
                    "ScaleStep":0.5,"NumWarps":5,"Epsilon":0.01,
                    "NumIterations":300}
            # args = {"Tau":0.25,"Lambda":0.2,"Theta":0.3,"NumScales":100,
            #         "ScaleStep":1,"NumWarps":5,"Epsilon":0.01,
            #         "NumIterations":300}
            frame_curr,frame_next = pair2gpu(frame_curr,frame_next)
            optical_flow = cv.cuda.OpticalFlowDual_TVL1_create()
            set_params(optical_flow,args)
            flow = optical_flow.calc(frame_curr, frame_next, None)
            return flow.download()
    elif fxn_s == "nofs":
        def wrapper(frame_curr,frame_next):
            H,W = frame_curr.shape[-3:-1]
            frame_curr,frame_next = pair2gpu(frame_curr,frame_next)
            perfPreset = cv.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_SLOW
            outSize = cv.cuda.NvidiaOpticalFlow_2_0_NV_OF_OUTPUT_VECTOR_GRID_SIZE_1
            params = {'perfPreset':perfPreset,'outputGridSize':outSize}
            optical_flow = cv.cuda.NvidiaOpticalFlow_2_0_create((W,H),**params)
            flow = optical_flow.calc(frame_curr, frame_next, None)[0]
            return flow.download()
    else:
        raise ValueError("Uknown Flow %s" % fxn_s)
    return wrapper

def set_params(fxn,args):
    for key,val in args.items():
        getattr(fxn,"set%s"%key)(val)

def get_flow_fxn_cpu(fxn_s):
    if fxn_s == "farne":
        def wrapper(frame_curr,frame_next):
            return cv.calcOpticalFlowFarneback(frame_curr,frame_next,flow=None,
                                               pyr_scale=0.5, levels=5,
                                               winsize=5,iterations=10,
                                               poly_n=5, poly_sigma=1.2,
                                               flags=10)
    elif fxn_s == "tvl1":
        def wrapper(frame_curr,frame_next):
            optical_flow = cv.optflow.DualTVL1OpticalFlow_create()
            flow = optical_flow.calc(frame_curr, frame_next, None)
            return flow
#define PAR_DEFAULT_OUTFLOW "flow.flo"
#define PAR_DEFAULT_NPROC   0
#define PAR_DEFAULT_TAU     0.25
#define PAR_DEFAULT_LAMBDA  0.15
#define PAR_DEFAULT_THETA   0.3
#define PAR_DEFAULT_NSCALES 100
#define PAR_DEFAULT_FSCALE  0
#define PAR_DEFAULT_ZFACTOR 0.5
#define PAR_DEFAULT_NWARPS  5
#define PAR_DEFAULT_EPSILON 0.01
#define PAR_DEFAULT_VERBOSE 0
    elif fxn_s == "tvl1p":
        def wrapper(frame_curr,frame_next):
            optical_flow = cv.optflow.DualTVL1OpticalFlow_create()
            args = {"Tau":0.25,"Lambda":0.15,"Theta":0.3,"ScalesNumber":5,
                    "ScaleStep":0.5,"WarpingsNumber":5,"Epsilon":0.01,
                    "InnerIterations":30,"OuterIterations":20,
                    "MedianFiltering":1}
            set_params(optical_flow,args)
            flow = optical_flow.calc(frame_curr, frame_next, None)
            return flow
    else:
        raise ValueError("Uknown Flow %s" % fxn_s)
    return wrapper

def pair2flow(frame_a,frame_b,flow_fxn,device):

    # -- numpy --
    frame_a = frame_a.cpu().numpy()
    frame_b = frame_b.cpu().numpy()

    # -- exec flow --
    flow = flow_fxn(frame_a,frame_b)

    # -- format flow --
    flow = flow.transpose(2,0,1)
    flow = th.from_numpy(flow).to(device)

    return flow

def pair2gpu(frame_a,frame_b):
    gpu_frame_a = cv.cuda_GpuMat()
    gpu_frame_b = cv.cuda_GpuMat()
    gpu_frame_a.upload(frame_a)
    gpu_frame_b.upload(frame_b)
    return gpu_frame_a,gpu_frame_b

def remove_batch(in_flows):
    if len(in_flows.fflow.shape) == 5:
        return index_at(in_flows,0,0)
    else:
        return in_flows

def index_at(in_flows,index,dim):
    if in_flows is None:
        return None
    out_flows = edict()
    if dim == 0:
        out_flows.fflow = in_flows.fflow[index]
        out_flows.bflow = in_flows.bflow[index]
    else:
        raise ValueError("Find a better way.")
    # index = th.LongTensor([index]).to(in_flows.fflow.device)
    # out_flows = edict()
    # out_flows.fflow = th.index_select(in_flows.fflow,dim,index)
    # out_flows.bflow = th.index_select(in_flows.bflow,dim,index)
    return out_flows

def slice_at(in_flows,dslice,dim):
    if in_flows is None:
        return None
    out_flows = edict()
    # fslice = [slice(None),]*(flows.fflow.ndim-dim) + [dslice,]
    if dim >= 0:
        fslice = [slice(None),]*dim + [dslice,]
    else:
        ndim = in_flows.fflow.ndim
        fslice = [slice(None),]*(ndim+dim) + [dslice,]
    out_flows.fflow = in_flows.fflow[fslice].contiguous()#.clone()
    out_flows.bflow = in_flows.bflow[fslice].contiguous()#.clone()
    return out_flows

def rslice(flows,region):
    _flows = edict()
    for key in flows:
        _flows[key] = rslice_tensor(flows[key],region)
    return _flows


def rescale_flows(flows_og,H,W):

    # -- corner case --
    if flows_og is None: return None

    # -- check --
    B,T,_,_H,_W = flows_og.fflow.shape
    if _H == H:
        return flows_og

    # -- alloc --
    fflow = flows_og.fflow.view(B*T,2,_H,_W)
    bflow = flows_og.bflow.view(B*T,2,_H,_W)
    shape = (H,W)

    # -- create new flows --
    flows = edict()
    flows.fflow = tnnf.interpolate(fflow,size=shape,mode="bilinear")
    flows.bflow = tnnf.interpolate(bflow,size=shape,mode="bilinear")

    # -- reshape --
    flows.fflow = flows.fflow.view(B,T,2,H,W)
    flows.bflow = flows.bflow.view(B,T,2,H,W)

    return flows

# """
# Wrap the opencv optical flow

# """

# # -- linalg --
# import numpy as np
# import torch as th
# from einops import rearrange,repeat

# # -- estimate sigma --
# from skimage.restoration import estimate_sigma

# # -- misc --
# from easydict import EasyDict as edict

# # -- opencv --
# import cv2 as cv

# # -- local --
# from ..utils import color

# def run_batch(vid,sigma):
#     B = vid.shape[0]
#     flows = edict()
#     flows.fflow,flows.bflow = [],[]
#     for b in range(B):
#         flows_b = run(vid[b],sigma)
#         flows.fflow.append(flows_b.fflow)
#         flows.bflow.append(flows_b.bflow)
#     flows.fflow = th.stack(flows.fflow)
#     flows.bflow = th.stack(flows.bflow)
#     return flows

# def run(vid_in,sigma):

#     # -- init --
#     device = vid_in.device
#     vid_in = vid_in.cpu()
#     vid = vid_in.clone() # copy data for no-rounding-error from RGB <-> YUV
#     t,c,h,w = vid.shape

#     # -- color2gray --
#     vid = th.clamp(vid,0,255.).type(th.uint8)
#     if vid.shape[1] == 3:
#         color.rgb2yuv(vid)
#     vid = vid[:,[0],:,:]
#     vid = rearrange(vid,'t c h w -> t h w c')

#     # -- alloc --
#     fflow = th.zeros((t,2,h,w),device=device)
#     bflow = th.zeros((t,2,h,w),device=device)

#     # -- computing --
#     for ti in range(t-1):
#         fflow[ti] = pair2flow(vid[ti],vid[ti+1],device)
#     for ti in reversed(range(t-1)):
#         bflow[ti+1] = pair2flow(vid[ti+1],vid[ti],device)

#     # -- final shaping --
#     # fflow = rearrange(fflow,'t h w c -> t c h w')
#     # bflow = rearrange(bflow,'t h w c -> t c h w')

#     # -- packing --
#     flows = edict()
#     flows.fflow = fflow
#     flows.bflow = bflow

#     # -- gray2color --
#     color.yuv2rgb(vid)

#     return flows

# def est_sigma(vid):
#     if vid.shape[-3] == 3:
#         vid = vid.cpu().clone()
#         color.rgb2yuv(vid)
#     vid_np = vid.cpu().numpy()
#     vid_np = vid_np[:,[0]] # Y only
#     sigma = estimate_sigma(vid_np,channel_axis=1)[0]
#     return sigma

# def pair2flow(frame_a,frame_b,device):
#     if "cpu" in str(frame_a.device):
#         return pair2flow_cpu(frame_a,frame_b,device)
#     else:
#         return pair2flow_gpu(frame_a,frame_b,device)

# def pair2flow_cpu(frame_a,frame_b,device):

#     # -- numpy --
#     frame_a = frame_a.cpu().numpy()
#     frame_b = frame_b.cpu().numpy()

#     # -- exec flow --
#     flow = cv.calcOpticalFlowFarneback(frame_a,frame_b,
#                                         0.,0.,3,15,3,5,1.,0)
#     flow = flow.transpose(2,0,1)
#     flow = th.from_numpy(flow).to(device)

#     return flow

# def pair2flow_gpu(frame_a,frame_b,device):

#     # -- create opencv-gpu frames --
#     gpu_frame_a = cv.cuda_GpuMat()
#     gpu_frame_b = cv.cuda_GpuMat()
#     gpu_frame_a.upload(frame_a.cpu().numpy())
#     gpu_frame_b.upload(frame_b.cpu().numpy())

#     # -- create flow object --
#     gpu_flow = cv.cuda_FarnebackOpticalFlow.create(5, 0.5, False,
#                                                    15, 3, 5, 1.2, 0)

#     # -- exec flow --
#     flow = cv.cuda_FarnebackOpticalFlow.calc(gpu_flow, gpu_frame_a,
#                                              gpu_frame_b, None)
#     flow = flow.download()
#     flow = flow.transpose(2,0,1)
#     flow = th.from_numpy(flow).to(device)

#     return flow
