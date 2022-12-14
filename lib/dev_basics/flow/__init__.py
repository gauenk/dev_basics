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

# -- local --
from ..utils import color

def run_zeros(vid,sigma=0.):
    device = vid.device
    b,t,c,h,w = vid.shape
    flows = edict()
    flows.fflow = th.zeros((b,t,2,h,w),device=device)
    flows.bflow = th.zeros((b,t,2,h,w),device=device)
    return flows

def orun(noisy,run_bool=True,sigma=None): # optional run
    if run_bool:
        if sigma is None:
            sigma_est = est_sigma(noisy)
        else:
            sigma_est = sigma
        if len(noisy.shape) == 5:
            flows = run_batch(noisy,sigma_est)
        else:
            flows = run_batch(noisy[None,:],sigma_est)
    else:
        if len(noisy.shape) == 5:
            flows = run_zeros(noisy)
        else:
            flows = run_zeros(noisy[None,:])
    return flows

def run_batch(vid,sigma,ftype="cv2"):
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

def run(vid_in,sigma,ftype="cv2"):
    if ftype == "cv2":
        return run_cv2(vid_in,sigma)
    elif ftype == "svnlb":
        return run_svnlb(vid_in,sigma)
    else:
        raise ValueError(f"Uknown flow type [{ftype}]")

#
# -- details --
#

def run_svnlb(vid_in,sigma):

    # -- run --
    vid_in_c = vid_in.cpu().numpy()
    fflow,bflow = svnlb.swig.runPyFlow(vid_in_c,sigma)

    # -- packing --
    flows = edict()
    flows.fflow = th.from_numpy(fflow).to(vid_in.device)
    flows.bflow = th.from_numpy(bflow).to(vid_in.device)
    return flows

def run_cv2(vid_in,sigma,rescale=True):

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
    for ti in range(t-1):
        fflow[ti] = pair2flow(vid[ti],vid[ti+1],device)
    for ti in reversed(range(t-1)):
        bflow[ti+1] = pair2flow(vid[ti+1],vid[ti],device)

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
    if vid.shape[1] == 3:
        vid = vid.cpu().clone()
        color.rgb2yuv(vid)
    vid_np = vid.cpu().numpy()
    vid_np = vid_np[:,[0]] # Y only
    sigma = estimate_sigma(vid_np,channel_axis=1)[0]
    return sigma

def pair2flow(frame_a,frame_b,device):
    if "cpu" in str(frame_a.device):
        return pair2flow_cpu(frame_a,frame_b,device)
    else:
        return pair2flow_gpu(frame_a,frame_b,device)

def pair2flow_cpu(frame_a,frame_b,device):

    # -- numpy --
    frame_a = frame_a.cpu().numpy()
    frame_b = frame_b.cpu().numpy()

    # -- exec flow --
    # flow = cv.calcOpticalFlowFarneback(frame_a,frame_b,
    #                                    0.,0.,3,15,3,5,1.,0)
    flow = cv.calcOpticalFlowFarneback(frame_a,frame_b,flow=None,
                                       pyr_scale=0.5, levels=5, winsize=5,
                                       iterations=10, poly_n=5, poly_sigma=1.2,
                                       flags=10)
    flow = flow.transpose(2,0,1)
    flow = th.from_numpy(flow).to(device)

    return flow

def pair2flow_gpu(frame_a,frame_b,device):

    # -- create opencv-gpu frames --
    gpu_frame_a = cv.cuda_GpuMat()
    gpu_frame_b = cv.cuda_GpuMat()
    gpu_frame_a.upload(frame_a.cpu().numpy())
    gpu_frame_b.upload(frame_b.cpu().numpy())

    # -- create flow object --
    gpu_flow = cv.cuda_FarnebackOpticalFlow.create(5, 0.5, False,
                                                   15, 3, 5, 1.2, 0)

    # -- exec flow --
    flow = cv.cuda_FarnebackOpticalFlow.calc(gpu_flow, gpu_frame_a,
                                             gpu_frame_b, None)
    flow = flow.download()
    flow = flow.transpose(2,0,1)
    flow = th.from_numpy(flow).to(device)

    return flow

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
