
from pathlib import Path

def get_pretrained_path_str(path):
    path = str(Path(path).stem)
    if "epoch" in path:
        epoch = path.split("=")[1].split(".")[0]
        uuid = path.split("-")[0][:4]
        pstr = "%s-%s" % (uuid,epoch)
    elif path == "none":
        pstr = "none"
    else:
        pstr = path
    return pstr

