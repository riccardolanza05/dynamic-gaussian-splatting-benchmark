import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def facet(metric, ylabel, title, name, mode="iterations", xcol="iter_total",
          xlabel="Training iterations (incl. coarse stage for 4DGaussians)",
          logy=False, invert=False, hline=None):
    fig, axes = plt.subplots(2,4, figsize=(13.5,6.4), sharex=True)
    for ax, s in zip(axes.ravel(), SCENES):
        for m in ORDER:
            d = cur(m,s,mode)
            if d.empty or d[metric].isna().all():
                continue
            p = is_partial(m,s,mode)
            ax.plot(d[xcol], d[metric], color=C[m], lw=2.0,
                    ls="--" if p else "-", alpha=0.95, zorder=3)
            if p:
                ax.scatter(d[xcol].iloc[-1], d[metric].iloc[-1], s=22, color=C[m],
                           marker="X", zorder=4, linewidths=0)
        ax.set_title(SCENE_LABEL[s], color=INK, loc="left", fontweight="bold")
        ax.grid(True, axis="both", alpha=0.9); ax.set_axisbelow(True)
        if logy: ax.set_yscale("log")
        if invert: ax.invert_yaxis()
        if hline is not None:
            ax.axhline(hline, color=MUTED, lw=0.8, ls=":", zorder=1)
        if (m:="4DGS-fudan") and (m,s) in NOT_EVAL:
            ax.text(0.5,0.5,"not evaluable\nfor 4DGS native-4D", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7.5, color=MUTED, style="italic")
        ax.xaxis.set_major_formatter(lambda v,p: f"{v/1000:g}k" if v>=1000 else f"{v:g}")
    for ax in axes[:,0]: ax.set_ylabel(ylabel)
    for ax in axes[1,:]: ax.set_xlabel(xlabel, fontsize=8)
    fig.suptitle(title, color=INK, fontsize=12.5, fontweight="bold", y=1.075, x=0.5)
    method_legend(fig, extra=[Line2D([],[],color=MUTED,lw=2,ls="--",label="partial run (stopped early)")], y=1.025)
    fig.tight_layout()
    savefig(fig, name, CAVEAT)

facet("psnr","PSNR (dB)","Reconstruction quality vs training iterations — D-NeRF monocular scenes",
      "01_psnr_vs_iterations", hline=30.0)
facet("ssim","SSIM","Structural similarity vs training iterations","02_ssim_vs_iterations")
facet("lpips","LPIPS (VGG)","Perceptual error vs training iterations (lower is better)",
      "03_lpips_vs_iterations", logy=True)
facet("eval_l1_loss","Eval L1 loss","Evaluation L1 loss vs training iterations (lower is better)",
      "04_l1_vs_iterations", logy=True)
facet("num_gaussians","# Gaussians","Number of Gaussian primitives vs training iterations",
      "05_gaussians_vs_iterations", logy=True)
facet("training_time_s","Training time (s)","Cumulative training time vs iterations (Tesla T4)",
      "06_traintime_vs_iterations")
facet("train_batch_loss","Train batch loss","Training batch loss vs iterations","07_trainloss_vs_iterations", logy=True)

# PSNR vs wall-clock training time
def facet_xy(xm, ym, xlabel, ylabel, title, name, mode="iterations", logx=False):
    fig, axes = plt.subplots(2,4, figsize=(13.5,6.4))
    for ax, s in zip(axes.ravel(), SCENES):
        for m in ORDER:
            d = cur(m,s,mode)
            if d.empty or d[ym].isna().all(): continue
            p = is_partial(m,s,mode)
            ax.plot(d[xm], d[ym], color=C[m], lw=2.0, ls="--" if p else "-", zorder=3)
        ax.set_title(SCENE_LABEL[s], color=INK, loc="left", fontweight="bold")
        ax.grid(True, alpha=0.9); ax.set_axisbelow(True)
        if logx: ax.set_xscale("log")
        if ("4DGS-fudan",s) in NOT_EVAL:
            ax.text(0.5,0.5,"not evaluable\nfor 4DGS native-4D", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7.5, color=MUTED, style="italic")
    for ax in axes[:,0]: ax.set_ylabel(ylabel)
    for ax in axes[1,:]: ax.set_xlabel(xlabel, fontsize=8)
    fig.suptitle(title, color=INK, fontsize=12.5, fontweight="bold", y=1.075)
    method_legend(fig, extra=[Line2D([],[],color=MUTED,lw=2,ls="--",label="partial run")], y=1.025)
    fig.tight_layout(); savefig(fig, name, CAVEAT)

facet_xy("training_time_s","psnr","Training time (s, Tesla T4)","PSNR (dB)",
         "Quality vs compute budget — PSNR against wall-clock training time","08_psnr_vs_traintime")
facet_xy("num_gaussians","psnr","# Gaussian primitives","PSNR (dB)",
         "Quality vs model size — PSNR against number of Gaussians","09_psnr_vs_gaussians", logx=True)
facet_xy("images_seen","psnr","Training images seen (batch-size corrected)","PSNR (dB)",
         "Quality vs number of training samples seen (fair across batch sizes)","10_psnr_vs_imagesseen")
