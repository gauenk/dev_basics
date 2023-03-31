"""

Evaluation classes

"""

# -- utils --
from dev_basics.utils import vid_io


def get_evaluator(test_task):
    if test_task == "deno":
        return DenoEvaluator()
    elif test_task in ["bbox","segm","keypoint"]:
        return SegEvaluator()
    else:
        raise ValueError(f"Uknown evaluator [{test_task}]")


class DenoEvaluator():

    def __init__(self):
        pass

    def get_keys(self):
        return ["noisy_psnrs","psnrs","ssims","strred","deno_fns"]

    def save_output(self,cfg,output):
        out_dir = Path(cfg.saved_dir) / cfg.arch_name / str(cfg.uuid)
        if tcfg.save_deno:
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
        tasks = ["bbox","segm"]
        out_dir = Path(cfg.saved_dir) / cfg.arch_name / str(cfg.uuid)
        self.coco_eval = COCOEvaluator(tasks,output_dir=out_dir)
        self.coco_eval.reset()

    def get_keys(self):
        return ["map"]

    def save_output(self,output):
        return {}

    def eval_ouptut(self,inputs,outputs):
        inputs['image_id'] = inputs['image_index']
        self.coco_eval.process(inputs,outputs)
        res = self.coco_eval.evaluate(inputs['image_id'])
        return res


