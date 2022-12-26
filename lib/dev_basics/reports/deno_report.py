
import numpy as np
import pandas as pd
from .shared import get_pretrained_path_str

def run(records,fields,fields_summ,res_fields,res_fmt):

    # -- init report --
    report = ""

    # -- run agg --
    agg = {key:[np.stack] for key in res_fields}
    grouped = records.groupby(fields).agg(agg)
    grouped.columns = res_fields
    grouped = grouped.reset_index()
    for field in res_fields:
        grouped[field] = grouped[field].apply(np.mean)
    res_fmt = {k:v for k,v in zip(res_fields,res_fmt)}
    report += "\n\n" + "-="*10+"-" + "Report" + "-="*10+"-" + "\n"
    for sigma,sdf in grouped.groupby("sigma"):
        report += ("--> sigma: %d <--" % sigma) + "\n"
        for gfields,gdf in sdf.groupby(fields_summ):
            gfields = list(gfields)

            # -- header --
            header = "-"*5 + " ("
            for i,field in enumerate(fields_summ):
                if field == "pretrained_path":
                    path_str = get_pretrained_path_str(gfields[i])
                    header += "%s, " % (path_str)
                else:
                    header += "%s, " % (gfields[i])
            header = header[:-2]
            header += ") " + "-"*5
            report += header + "\n"
            # print(header)

            for vid_name,vdf in gdf.groupby("vid_name"):
                # -- res --
                res = "%13s: " % vid_name
                for field in res_fields:
                    res += res_fmt[field] % (vdf[field].mean()) + " "
                report += res + "\n"

            # -- res --
            res = "%13s: " % "Ave"
            for field in res_fields:
                res += res_fmt[field] % (gdf[field].mean()) + " "
            report += res + "\n"
    return report

