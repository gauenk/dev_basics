import torch as th
import numpy as np
from ..common import optional


"""

  Handle Super-Resolution by Allocating after seeing the output dimensions

"""

def get_outputs(deno,Z,rH,rW,vid_out_chunk,vid_in_chunk,vid_in_full):
    if deno is None:
        return allocate_outputs(deno,Z,vid_out_chunk,vid_in_chunk,vid_in_full)
    else:
        return deno,Z,rH,rW

def allocate_outputs(deno,Z,vid_out_chunk,vid_in_chunk,vid_in_full):
    """

    Allocate output shape depending on the network's output shape.

    """
    device = vid_out_chunk.device
    C = vid_out_chunk.shape[-3]
    H,W,rH,rW = get_output_spatial_size(vid_out_chunk,vid_in_chunk,vid_in_full)
    oshape = list(vid_in_full.shape[:-3]) + [C,H,W]
    if deno is None:
        deno = th.zeros(oshape,device=device)
    if Z is None:
        Z = th.zeros((H,W),device=device)
    return deno,Z,rH,rW

def get_output_spatial_size(vid_out_chunk,vid_in_chunk,vid_in_full):
    outH,outW = vid_out_chunk.shape[-2:]
    inH,inW = vid_in_chunk.shape[-2:]
    rH,rW = int(outH/(1.*inH)),int(outW/(1.*inW))
    H,W = vid_in_full.shape[-2:]
    outH,outW = rH*H,rW*W
    return outH,outW,rH,rW

"""

  Misc

"""

def expand_match(vshape,tensor,dim):

    # -- append singletons --
    e1 = (1,) * (len(vshape)-1)
    e1 += (vshape[dim],)
    tensor = tensor.expand(*e1)

    # -- view --
    e2 = (1,) * len(vshape[:dim])
    e2 += (vshape[dim],)
    e2 += (1,) * len(vshape[dim+1:])
    tensor = tensor.view(e2)

    return tensor

def expand2square(timg,factor=16.0):
    b, t, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = th.zeros(b,t,3,X,X).type_as(timg) # 3, h,w
    mask = th.zeros(b,t,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)

    return img, mask

def get_chunks(size,chunk_size,overlap):
    """

    Thank you to https://github.com/Devyanshu/image-split-with-overlap/

    args:
      size = original size
      chunk_size = size of output chunks
      overlap = percent (from 0.0 - 1.0) of overlap for each chunk

    This code splits an input size into chunks to be used for
    split processing

    """
    overlap = handle_int_overlap(size,overlap)
    points = [0]
    stride = max(int(chunk_size * (1-overlap)),1)
    if size <= chunk_size: return [0]
    assert stride > 0
    counter = 1
    while True:
        pt = stride * counter
        if pt + chunk_size >= size:
            points.append(size - chunk_size)
            break
        else:
            points.append(pt)
        counter += 1
    points = list(np.unique(points))
    return points


def handle_int_overlap(size,overlap):
    if overlap >= 0 and overlap < 1:
        return overlap
    elif overlap >= 1:
        if isinstance(overlap,int):
            return (1.*overlap)/size
    else:
        raise ValueError("Uknown behavior for overlap as a float greater than 1 or less than 0.")

def expected_runs(cfg,T,C,H,W):

    # -- unpack --
    t_size = optional(cfg,'temporal_chunk_size',0)
    t_over = optional(cfg,'temporal_chunk_overlap',0)
    c_size = optional(cfg,'channel_chunk_size',0)
    c_over = optional(cfg,'channel_chunk_overlap',0)
    s_size = optional(cfg,'spatial_chunk_size',0)
    s_over = optional(cfg,'spatial_chunk_overlap',0)

    # # -- expected runs --
    # t_runs = 0
    # if t_size > 0:
    #     stride = t_size * t_over
    #     t_runs = (T-1)//tsize
    # T,C,H,W
    return 0
