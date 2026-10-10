import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
R=runs.copy()
IT=R[R["mode"]=="iterations"]; LO=R[R["mode"]=="target_eval_loss"]
from study import FAIR   # scenes completed by every method of the active study

t1=IT[["method","scene","partial","best_psnr","best_psnr_iter","best_ssim","best_lpips",
       "best_eval_l1_loss","final_psnr","final_total_iterations","final_training_time_s",
       "final_num_gaussians","peak_vram_mb","process_wall_time_s"]].copy()
t1["psnr_drop_peak_to_end"]=t1.best_psnr-t1.final_psnr
t1["s_per_1k_iters"]=1000*t1.final_training_time_s/t1.final_total_iterations
t1=t1.merge(IT[["method","scene","batch_size","final_images_seen"]],on=["method","scene"],how="left")
t1=t1.sort_values(["scene","method"])
t1.to_csv(os.path.join(TAB,"table_iso_iterations.csv"),index=False)

t2=LO[["method","scene","batch_size","target_eval_loss","fc_total_iterations","fc_images_seen",
       "fc_training_time_s","fc_psnr","fc_ssim","fc_lpips","fc_num_gaussians",
       "final_total_iterations","final_training_time_s","best_psnr","final_psnr","final_ssim",
       "final_lpips","final_eval_l1_loss","final_num_gaussians","model_storage_mb","full_folder_mb",
       "peak_vram_mb","status","target_reached"]].sort_values(["scene","method"])
t2.to_csv(os.path.join(TAB,"table_iso_loss.csv"),index=False)

agg=[]
for m in ORDER:
    a=IT[(IT.method_short==m)&(IT.scene.isin(FAIR))]; b=LO[(LO.method_short==m)&(LO.scene.isin(FAIR))]
    agg.append(dict(method=LABEL[m],
        psnr_mean=a.best_psnr.mean(), ssim_mean=a.best_ssim.mean(), lpips_mean=a.best_lpips.mean(),
        l1_mean=a.best_eval_l1_loss.mean(),
        psnr_drop_mean=(a.best_psnr-a.final_psnr).mean(),
        iter_at_peak_mean=a.best_psnr_iter.mean(),
        train_time_30k_min=a.final_training_time_s.mean()/60,
        s_per_1k=(1000*a.final_training_time_s/a.final_total_iterations).mean(),
        vram_mean_mb=a.peak_vram_mb.mean(), gauss_final_mean=a.final_num_gaussians.mean(),
        iters_to_target_mean=b.fc_total_iterations.mean(),
        images_to_target_mean=b.fc_images_seen.mean(),
        time_to_target_min=b.fc_training_time_s.mean()/60,
        psnr_at_target_mean=b.fc_psnr.mean(), storage_at_target_mb=b.model_storage_mb.mean(),
        gauss_at_target_mean=b.fc_num_gaussians.mean()))
agg=pd.DataFrame(agg)
if STUDY!="monocular":
    # Protocol A read where the authors read it (per-scene snapshot for Spacetime Gaussians,
    # the end of the official budget for the others), next to the end and the best of the run.
    t3=IT[["method","scene","official_iteration","official_psnr","official_ssim","official_lpips",
           "official_lpips_alex","final_lpips_alex","best_lpips_alex",
           "official_eval_l1_loss","official_training_time_s","official_images_seen",
           "final_iteration","final_psnr","final_ssim","final_lpips","final_training_time_s",
           "best_psnr","best_psnr_iter","peak_vram_mb","model_storage_mb",
           "preparation_time_s","scene_data_mb"]].sort_values(["scene","method"])
    t3.to_csv(os.path.join(TAB,"table_official_readout.csv"),index=False)
    extra=[]
    for m in ORDER:
        a=IT[(IT.method_short==m)&(IT.scene.isin(FAIR))]
        extra.append(dict(psnr_official_mean=a.official_psnr.mean(),
                          ssim_official_mean=a.official_ssim.mean(),
                          lpips_official_mean=a.official_lpips.mean(),
                          lpips_alex_official_mean=a.official_lpips_alex.mean(),
                          lpips_alex_best_mean=a.best_lpips_alex.mean(),
                          time_official_min=a.official_training_time_s.mean()/60,
                          psnr_final_mean=a.final_psnr.mean(),
                          storage_mb_mean=a.model_storage_mb.mean()))
    agg=pd.concat([agg,pd.DataFrame(extra)],axis=1)
agg.to_csv(os.path.join(TAB,"table_aggregates_fair5.csv"),index=False)
pd.set_option("display.width",250)
print(agg.round(3).to_string(index=False))
print()
print(t2[["method","scene","batch_size","fc_total_iterations","fc_images_seen","fc_training_time_s","fc_psnr","fc_lpips","model_storage_mb"]].round(3).to_string(index=False))
