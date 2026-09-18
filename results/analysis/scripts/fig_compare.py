import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

R = runs.copy()
IT = R[R["mode"]=="iterations"].set_index(["method_short","scene"])
LO = R[R["mode"]=="target_eval_loss"].set_index(["method_short","scene"])

def val(tab, m, s, col):
    try: return float(tab.loc[(m,s), col])
    except Exception: return np.nan

def grouped_bar(getter, ylabel, title, name, mode="iterations", fmt="{:.2f}",
                lower_better=False, logy=False, note_extra="", scenes=SCENES, annotate=True):
    x = np.arange(len(scenes)); w = 0.26
    fig, ax = plt.subplots(figsize=(12.6,4.2))
    nanpos = []
    for i,m in enumerate(ORDER):
        vals = [getter(m,s) for s in scenes]
        pos  = x + (i-1)*w
        hatch = ["///" if is_partial(m,s,mode) else "" for s in scenes]
        for xi,v,h,s in zip(pos, vals, hatch, scenes):
            if np.isnan(v):
                nanpos.append(xi)
                continue
            ax.bar(xi, v, w*0.92, color=C[m], edgecolor=SURF, linewidth=1.2,
                   hatch=h, zorder=3)
            if annotate:
                ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=6.8,
                        color=INK2, rotation=90 if len(fmt.format(v))>5 else 0,
                        clip_on=False)
    ax.set_xticks(x); ax.set_xticklabels([SCENE_LABEL[s] for s in scenes])
    ax.tick_params(axis="x", colors=INK2, labelsize=8.5)
    ax.set_ylabel(ylabel); ax.grid(True, axis="y", alpha=0.9); ax.set_axisbelow(True)
    if logy: ax.set_yscale("log")
    ax.margins(y=0.16)
    y0,y1 = ax.get_ylim()
    for xi in nanpos:
        ax.text(xi, y0 + 0.02*(y1-y0), "n/a", ha="center", va="bottom", fontsize=8,
                color=MUTED, rotation=90, style="italic")
    fig.suptitle(title + ("  (lower is better)" if lower_better else "  (higher is better)"),
                 color=INK, fontsize=12.5, fontweight="bold", x=0.012, ha="left", y=1.115)
    method_legend(fig, extra=[Patch(facecolor=SURF, edgecolor=MUTED, hatch="///", label="partial run")],
                  y=1.045)
    fig.tight_layout(); savefig(fig, name, CAVEAT+" "+note_extra)

# ---------- ISO-ITERATIONS: best metric of each run ----------
grouped_bar(lambda m,s: val(IT,m,s,"best_psnr"), "Best PSNR (dB)",
    "Best PSNR per scene — equal-iteration budget (30k total iterations)",
    "11_iso_iters_best_psnr", fmt="{:.2f}")
grouped_bar(lambda m,s: val(IT,m,s,"best_ssim"), "Best SSIM",
    "Best SSIM per scene — equal-iteration budget", "12_iso_iters_best_ssim", fmt="{:.4f}")
grouped_bar(lambda m,s: val(IT,m,s,"best_lpips"), "Best LPIPS (VGG)",
    "Best LPIPS per scene — equal-iteration budget", "13_iso_iters_best_lpips",
    fmt="{:.4f}", lower_better=True)
grouped_bar(lambda m,s: val(IT,m,s,"best_eval_l1_loss"), "Best eval L1 loss",
    "Best evaluation L1 loss per scene — equal-iteration budget", "14_iso_iters_best_l1",
    fmt="{:.4f}", lower_better=True)
grouped_bar(lambda m,s: val(IT,m,s,"best_psnr_iter"), "Iteration of best PSNR",
    "Iteration at which the best PSNR is reached — equal-iteration budget",
    "15_iso_iters_argmax_psnr", fmt="{:.0f}", lower_better=True)

# ---------- ISO-LOSS: run stops at the same per-scene L1 target ----------
BSNOTE = ("Measured at the first crossing of the target, not at run termination. "
          "4DGS native-4D uses the per-scene batch sizes of its own repository (1-24), so its iteration "
          "counts are not directly comparable: see the training-samples figure (38).")
grouped_bar(lambda m,s: val(LO,m,s,"fc_total_iterations"), "Iterations to target",
    "Optimisation steps needed to reach the common per-scene L1 target", "16_iso_loss_iters_to_target",
    mode="target_eval_loss", fmt="{:.0f}", lower_better=True, note_extra=BSNOTE)
grouped_bar(lambda m,s: val(LO,m,s,"fc_training_time_s"), "Training time to target (s)",
    "Wall-clock training time to reach the common per-scene L1 target (Tesla T4)",
    "17_iso_loss_time_to_target", mode="target_eval_loss", fmt="{:.0f}", lower_better=True, note_extra=BSNOTE)
grouped_bar(lambda m,s: val(LO,m,s,"fc_psnr"), "PSNR at target (dB)",
    "PSNR at the first crossing of the common per-scene L1 target",
    "18_iso_loss_psnr_at_target", mode="target_eval_loss", fmt="{:.2f}", note_extra=BSNOTE)
grouped_bar(lambda m,s: val(LO,m,s,"fc_images_seen"), "Training images seen",
    "Training samples consumed to reach the common per-scene L1 target (batch-size corrected)",
    "38_iso_loss_images_to_target", mode="target_eval_loss", fmt="{:.0f}", lower_better=True,
    annotate=False, note_extra=BSNOTE)
grouped_bar(lambda m,s: val(LO,m,s,"fc_num_gaussians"), "# Gaussians at target",
    "Model size (number of Gaussians) at the common per-scene L1 target",
    "19_iso_loss_gaussians_at_target", mode="target_eval_loss", fmt="{:.0f}",
    lower_better=True, logy=True, annotate=False)
grouped_bar(lambda m,s: val(LO,m,s,"model_storage_mb"), "Model storage (MB)",
    "On-disk model size at the common per-scene L1 target",
    "20_iso_loss_storage_at_target", mode="target_eval_loss", fmt="{:.1f}", lower_better=True)
grouped_bar(lambda m,s: val(LO,m,s,"fc_lpips"), "LPIPS at target",
    "LPIPS at the common per-scene L1 target", "21_iso_loss_lpips_at_target",
    mode="target_eval_loss", fmt="{:.4f}", lower_better=True)

# ---------- cost ----------
grouped_bar(lambda m,s: val(IT,m,s,"peak_vram_mb"), "Peak VRAM (MB)",
    "Peak GPU memory during the equal-iteration runs (nvidia-smi, Tesla T4)",
    "22_peak_vram_iters", fmt="{:.0f}", lower_better=True)
grouped_bar(lambda m,s: (val(IT,m,s,"final_training_time_s")/max(val(IT,m,s,"final_total_iterations"),1))*1000,
    "Seconds per 1000 iterations", "Training throughput — seconds per 1000 iterations (Tesla T4)",
    "23_throughput_s_per_1k", fmt="{:.1f}", lower_better=True)
grouped_bar(lambda m,s: 100*val(IT,m,s,"final_benchmark_overhead_s")/max(val(IT,m,s,"final_wall_time_s"),1e-9),
    "Benchmark overhead (% of wall time)", "Cost of the periodic evaluation inside the run",
    "24_eval_overhead_pct", fmt="{:.1f}", lower_better=True)
