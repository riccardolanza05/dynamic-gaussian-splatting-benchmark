import json, os, glob
import pandas as pd, numpy as np

ROOT = os.environ.get("GS_ROOT") or os.path.dirname(os.path.dirname(
           os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(ROOT, "analysis")
os.makedirs(os.path.join(OUT,"tables"), exist_ok=True)

METHOD_DIR = {
 "4dgaussian_output": "4DGaussians (Wu et al.)",
 "4dgs_fudan_output": "4DGS native-4D (fudan-zvg)",
 "deformablegaussian": "Deformable-3DGS (Yang et al.)",
}
SHORT = {
 "4DGaussians (Wu et al.)": "4DGaussians",
 "4DGS native-4D (fudan-zvg)": "4DGS-fudan",
 "Deformable-3DGS (Yang et al.)": "Deformable-3DGS",
}

rows_c, rows_r = [], []
for mdir, mname in METHOD_DIR.items():
    for run in sorted(glob.glob(os.path.join(ROOT, mdir, "*"))):
        b = os.path.join(run, "benchmark")
        jfs = [f for f in glob.glob(os.path.join(b, "benchmark_*.json")) if "config" not in os.path.basename(f)]
        runname = os.path.basename(run)
        partial = runname.endswith("_partial")
        if not jfs:
            rows_r.append(dict(method=mname, method_short=SHORT[mname], run=runname, partial=partial,
                               missing=True))
            continue
        d = json.load(open(jfs[0]))
        cfg = d.get("config", {})
        scene = d.get("scene") or cfg.get("scene")
        mode  = d.get("mode")  or cfg.get("mode")
        ents  = d.get("entries") or []
        st    = d.get("storage") or {}
        off   = cfg.get("iteration_offset", 0) or 0
        for e in ents:
            r = dict(method=mname, method_short=SHORT[mname], scene=scene, mode=mode,
                     run=runname, partial=partial, iteration_offset=off)
            r.update({k: e.get(k) for k in
                      ["iteration","total_iterations","images_seen","stage","training_time_s",
                       "wall_time_s","benchmark_overhead_s","psnr","ssim","lpips","eval_l1_loss",
                       "eval_photometric_loss","eval_loss","num_gaussians","train_batch_loss",
                       "eval_duration_s"]})
            rows_c.append(r)
        fin = d.get("final") or {}
        best_psnr = d.get("best_psnr")
        if best_psnr is None and ents:
            best_psnr = max((e.get("psnr") or -np.inf) for e in ents)
        rr = dict(method=mname, method_short=SHORT[mname], scene=scene, mode=mode, run=runname,
                  partial=partial, missing=len(ents)==0,
                  status=d.get("status"), target_reached=d.get("target_reached"),
                  n_entries=len(ents),
                  best_psnr=best_psnr, best_psnr_iteration=d.get("best_psnr_iteration"),
                  process_wall_time_s=d.get("process_wall_time_s"),
                  peak_vram_mb=d.get("peak_vram_mb_nvidia_smi"),
                  model_storage_mb=st.get("model_storage_mb"),
                  full_folder_mb=st.get("full_folder_mb"),
                  target_eval_loss=cfg.get("target_eval_loss"),
                  max_iterations=cfg.get("max_iterations"),
                  batch_size=cfg.get("batch_size"), iteration_offset=off,
                  eval_every_n_iters=cfg.get("eval_every_n_iters"),
                  eval_every_n_seconds=cfg.get("eval_every_n_seconds"))
        for k in ["iteration","total_iterations","training_time_s","wall_time_s","benchmark_overhead_s",
                  "psnr","ssim","lpips","eval_l1_loss","num_gaussians"]:
            rr["final_"+k] = fin.get(k)
        # first crossing of the L1 target ("primo attraversamento")
        if mode=="target_eval_loss" and ents:
            tgt = cfg.get("target_eval_loss")
            hit = next((e for e in ents if e.get("eval_l1_loss") is not None and e["eval_l1_loss"]<=tgt), None)
            if hit:
                for k,name in [("iteration","fc_iteration"),("total_iterations","fc_total_iterations"),
                               ("training_time_s","fc_training_time_s"),("wall_time_s","fc_wall_time_s"),
                               ("images_seen","fc_images_seen"),("psnr","fc_psnr"),("ssim","fc_ssim"),
                               ("lpips","fc_lpips"),("eval_l1_loss","fc_eval_l1_loss"),
                               ("num_gaussians","fc_num_gaussians")]:
                    rr[name]=hit.get(k)
        # images seen at the end of the run
        if ents: rr["final_images_seen"]=ents[-1].get("images_seen")
        # best-by-metric rows (over the whole run)
        if ents:
            df = pd.DataFrame(ents)
            for m,better in [("psnr","max"),("ssim","max"),("lpips","min"),("eval_l1_loss","min")]:
                idx = df[m].idxmax() if better=="max" else df[m].idxmin()
                rr["best_"+m] = df.loc[idx,m]
                rr["best_"+m+"_iter"] = df.loc[idx,"iteration"]
                rr["best_"+m+"_time_s"] = df.loc[idx,"training_time_s"]
                rr["best_"+m+"_ngauss"] = df.loc[idx,"num_gaussians"]
        rows_r.append(rr)

curves = pd.DataFrame(rows_c)
runs   = pd.DataFrame(rows_r)
curves.to_csv(os.path.join(OUT,"tables","curves_all.csv"), index=False)
runs.to_csv(os.path.join(OUT,"tables","runs_summary.csv"), index=False)
print("curves", curves.shape, "runs", runs.shape)
print(runs[["method_short","scene","mode","status","target_reached","n_entries","best_psnr","best_psnr_iteration","final_iteration","final_psnr","peak_vram_mb","model_storage_mb","process_wall_time_s"]].to_string())
