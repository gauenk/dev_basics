"""

Evaluation classes

"""

# -- utils --
from dev_basics.utils import vid_io
from pathlib import Path
try:
    import detectron2
    from detectron2.evaluation import COCOEvaluator
except:
    pass

def get_evaluator(cfg,test_task):
    if test_task == "deno":
        return DenoEvaluator()
    elif test_task in ["bbox","seg","segm","keypoint"]:
        return SegEvaluator(cfg)
    else:
        raise ValueError(f"Uknown evaluator [{test_task}]")

class DenoEvaluator():

    def __init__(self):
        pass

    def get_keys(self):
        return ["noisy_psnrs","psnrs","ssims","strred","deno_fns"]

    def save_output(self,cfg,output):
        out_dir = Path(cfg.saved_dir) / cfg.arch_name / str(cfg.uuid)
        if cfg.save_deno:
            print("Saving Denoised Output [%s]" % out_dir)
            deno_fns = vid_io.save_video(deno,out_dir,"deno")
        else:
            deno_fns = ["" for _ in range(deno.shape[0])]
        return {"deno_fns":deno_fns}

    def eval_output(self,inputs,outputs):
        imax = 255.
        noisy = inputs['noisy']
        clean = inputs['clean']
        deno = outputs
        noisy_psnrs = compute_psnrs(noisy,clean,div=imax)
        psnrs = compute_psnrs(clean,deno,div=imax)
        ssims = compute_ssims(clean,deno,div=imax)
        strred = compute_strred(clean,deno,div=imax)
        return {"noisy_psnrs":noisy_psnrs,
                "psnrs":psnrs,"ssims":ssims,
                "strred":strred}

class SegEvaluator():

    def __init__(self,cfg):
        tasks = ("bbox","segm")
        print(cfg)
        dataset_name = "coco_2017_val"
        out_dir = Path(cfg.saved_dir) / cfg.arch_name / str(cfg.uuid)
        self.coco_eval = COCOEvaluator(dataset_name,output_dir=out_dir)
        self.coco_eval.reset()

    def get_keys(self):
        return ["map"]

    def save_output(self,output):
        return {}

    def eval_output(self,inputs,outputs):
        print(list(inputs.keys()))
        inputs['image_id'] = int(inputs['index'])
        # inputs = dictOfLists_to_listOfDicts(inputs)
        inputs = [inputs]
        self.coco_eval.process(inputs,outputs)
        res = self.coco_eval.evaluate(inputs[0]['image_id'])
        return res


def dictOfLists_to_listOfDicts(pydict):
    pylist = []
    L = len(pydict[list(pydict.keys())[0]])
    for i in range(L):
        pydict_i = {}
        for key in pydict:
            pydict_i[key] = pydict[key][i]
        pylist.append(pydict_i)
    return pylist
