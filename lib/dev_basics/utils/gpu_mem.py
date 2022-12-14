import torch as th

class GpuMemer(): # like "Timer"

    def __init__(self,use_mem=True):
        self.use_mem = use_mem
        self.mems_res = []
        self.mems_alloc = []
        self.names = []

    def __str__(self):
        msg = "--- GPU Mem ---"
        for k,v in self.items():
            msg += "\n%s: %2.3f,%2.3f\n" % (k,v[0],v[1])
        return msg

    def __getitem__(self,name):
        idx = self.names.index(name)
        mem_res = self.mems_res[idx]
        mem_alloc = self.mems_alloc[idx]
        mems = {"res":mem_res,"alloc":mem_alloc}
        return mems

    def __setitem__(self,name,mems):
        idx = self.names.index(name)
        self.mems_res[idx] = mems[0]
        self.mems_alloc[idx] = mems[1]

    def items(self):
        names = ["%s" % name for name in self.names]
        mems = zip(self.mems_res,self.mems_alloc)
        return zip(names,mems)

    def start(self,name):
        if self.use_mem is False: return
        th.cuda.empty_cache()
        th.cuda.reset_max_memory_allocated()
        th.cuda.synchronize()

    def stop(self,name):
        if self.use_mem is False: return
        th.cuda.synchronize()
        mem_alloc = th.cuda.memory_allocated() / 1024**3
        mem_res = th.cuda.memory_reserved() / 1024**3
        self.names.append(name)
        self.mems_alloc.append(mem_alloc)
        self.mems_res.append(mem_res)

class MemIt(): # like "TimeIt"

    def __init__(self,memer,name):
        self.memer = memer
        self.name = name

    def __enter__(self):
        """Start a new memer as a context manager"""
        self.memer.start(self.name)
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager memer"""
        self.memer.stop(self.name)

def print_gpu_stats(verbose,name):
    fmt_all = "[%s] Memory Allocated [GB]: %2.3f"
    fmt_res = "[%s] Memory Reserved [GB]: %2.3f"
    th.cuda.empty_cache()
    th.cuda.synchronize()
    mem_alloc = th.cuda.memory_allocated() / 1024**3
    mem_res = th.cuda.memory_reserved() / 1024**3
    if verbose:
        print(fmt_all % (name,mem_alloc))
        print(fmt_res % (name,mem_res))
    return mem_alloc,mem_res

def reset_peak_gpu_stats():
    th.cuda.reset_max_memory_allocated()

def print_peak_gpu_stats(verbose,name,reset=True):
    fmt = "[%s] Peak Memory Allocated [GB]: %2.3f"
    mem_alloc = th.cuda.max_memory_allocated(0) / (1024.**3)
    mem_res = th.cuda.max_memory_reserved(0) / (1024.**3)
    if verbose:
        th.cuda.empty_cache()
        th.cuda.synchronize()
        print(fmt % (name,mem_alloc))
    if reset: th.cuda.reset_peak_memory_stats()
    return mem_alloc,mem_res
