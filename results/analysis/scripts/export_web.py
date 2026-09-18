import sys,os,json; sys.path.insert(0,os.path.expanduser("~/gsa"))
from common import *
def clean(v):
    if v is None: return None
    try:
        f=float(v)
        return None if np.isnan(f) else (round(f,6) if abs(f)<1e4 else round(f,2))
    except Exception: return v
data={"methods":[{"key":m,"label":LABEL[m],"color":C[m]} for m in ORDER],
      "scenes":[{"key":s,"label":SCENE_LABEL[s]} for s in SCENES],
      "curves":{}, "runs":[]}
for m in ORDER:
    for s in SCENES:
        for mode in ["iterations","target_eval_loss"]:
            d=cur(m,s,mode)
            if d.empty: continue
            data["curves"][f"{m}|{s}|{mode}"]={
              "iter":[clean(x) for x in d.iter_total],
              "img":[clean(x) for x in d.images_seen],
              "time":[clean(x) for x in d.training_time_s],
              "psnr":[clean(x) for x in d.psnr],"ssim":[clean(x) for x in d.ssim],
              "lpips":[clean(x) for x in d.lpips],"l1":[clean(x) for x in d.eval_l1_loss],
              "ngauss":[clean(x) for x in d.num_gaussians],
              "partial":bool(is_partial(m,s,mode))}
cols=["method_short","scene","mode","status","target_reached","partial","missing","best_psnr",
      "best_psnr_iter","best_ssim","best_lpips","best_eval_l1_loss","final_psnr","final_ssim",
      "final_lpips","final_eval_l1_loss","final_total_iterations","final_training_time_s",
      "final_num_gaussians","peak_vram_mb","model_storage_mb","target_eval_loss","process_wall_time_s",
      "batch_size","final_images_seen","fc_total_iterations","fc_images_seen","fc_training_time_s",
      "fc_psnr","fc_ssim","fc_lpips","fc_num_gaussians"]
for _,r in runs.iterrows():
    data["runs"].append({c:(clean(r[c]) if c not in("method_short","scene","mode","status") else r[c]) for c in cols if c in runs.columns})
p=os.path.join(OUT,"dashboard_data.json")
json.dump(data,open(p,"w"),separators=(",",":"),default=str)
print(p, os.path.getsize(p))
