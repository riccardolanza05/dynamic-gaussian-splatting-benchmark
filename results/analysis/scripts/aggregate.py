import json, os, glob, re
import pandas as pd, numpy as np

ROOT = os.environ.get("GS_ROOT") or os.path.dirname(os.path.dirname(
           os.path.dirname(os.path.abspath(__file__))))
from study import METHOD_DIR, SHORT, STUDY, ANALYSIS_DIR, NUM_FRAMES, VARIANTS

OUT  = os.path.join(ROOT, ANALYSIS_DIR)
os.makedirs(os.path.join(OUT,"tables"), exist_ok=True)

print("study: %s" % STUDY)


def method_label(mdir, default, cfg, storage, runname):
    """Long label of a run. A folder listed in VARIANTS holds several variants of one
    method (Spacetime Gaussians lite and full), each reported as its own method: the
    variant is read from the benchmark config, then from the storage report, and a run
    with no JSON at all falls back on the tag in its folder name."""
    if mdir not in VARIANTS:
        return default
    key, labels, fallback = VARIANTS[mdir]
    value = cfg.get(key) or storage.get(key)
    if value is None:
        value = next((v for v in labels if v != fallback
                      and runname.endswith("_" + v.split("_")[-1])), fallback)
    return labels.get(value, default)


def official_iteration(cfg):
    """The step at which the authors read their own result on this scene, when the run
    recorded it: the snapshot Spacetime Gaussians' test.py evaluates (per scene), or the
    official budget of the other methods."""
    for key in ("official_test_iteration", "official_iterations"):
        if cfg.get(key) is not None:
            return int(cfg[key])
    return None

rows_c, rows_r = [], []
for mdir, mname in METHOD_DIR.items():
    for run in sorted(glob.glob(os.path.join(ROOT, mdir, "*"))):
        # A run folder carries its window in its name (<scene>_f<frames>_...): skip the other
        # study's runs before looking for a JSON, so they are not reported as missing either.
        _w = re.search(r"_f(\d+)_", os.path.basename(run))
        if NUM_FRAMES is not None and _w and int(_w.group(1)) != NUM_FRAMES:
            continue
        b = os.path.join(run, "benchmark")
        jfs = [f for f in glob.glob(os.path.join(b, "benchmark_*.json")) if "config" not in os.path.basename(f)]
        runname = os.path.basename(run)
        partial = runname.endswith("_partial")
        if not jfs:
            mlabel = method_label(mdir, mname, {}, {}, runname)
            rows_r.append(dict(method=mlabel, method_short=SHORT[mlabel], run=runname, partial=partial,
                               missing=True))
            continue
        d = json.load(open(jfs[0]))
        cfg = d.get("config", {})
        mname = method_label(mdir, METHOD_DIR[mdir], cfg, d.get("storage") or {}, runname)
        # The two N3DV studies share the method folders: keep only this study's window.
        if NUM_FRAMES is not None and (d.get("num_frames") or cfg.get("num_frames")) != NUM_FRAMES:
            continue
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
            # A second LPIPS backbone, recorded by the multi-view notebooks only.
            if STUDY != "monocular":
                r["lpips_alex"] = e.get("lpips_alex")
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
        # The authors' own reading of this run, next to its end and its best (Protocol A).
        if STUDY != "monocular":
            off = official_iteration(cfg) if mode == "iterations" else None
            rr["official_iteration"] = off
            hit = next((e for e in ents if e.get("iteration") == off), None) if off else None
            rr["final_lpips_alex"] = fin.get("lpips_alex")
            _alex = [e["lpips_alex"] for e in ents if e.get("lpips_alex") is not None]
            rr["best_lpips_alex"] = min(_alex) if _alex else None
            for k in ["psnr", "ssim", "lpips", "lpips_alex", "eval_l1_loss", "training_time_s",
                      "images_seen", "num_gaussians"]:
                rr["official_" + k] = hit.get(k) if hit else None
            rr["preparation_time_s"] = d.get("preparation_time_s")
            rr["scene_data_mb"] = d.get("scene_data_mb")
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
# A study with no Protocol B run (the 300-frame N3DV study has none by design) still gets
# the first-crossing columns, empty, so the scripts downstream find the schema they expect.
for _c in ["fc_iteration","fc_total_iterations","fc_training_time_s","fc_wall_time_s",
           "fc_images_seen","fc_psnr","fc_ssim","fc_lpips","fc_eval_l1_loss","fc_num_gaussians"]:
    if _c not in runs.columns:
        runs[_c] = np.nan
curves.to_csv(os.path.join(OUT,"tables","curves_all.csv"), index=False)
runs.to_csv(os.path.join(OUT,"tables","runs_summary.csv"), index=False)
print("curves", curves.shape, "runs", runs.shape)
print(runs[["method_short","scene","mode","status","target_reached","n_entries","best_psnr","best_psnr_iteration","final_iteration","final_psnr","peak_vram_mb","model_storage_mb","process_wall_time_s"]].to_string())
