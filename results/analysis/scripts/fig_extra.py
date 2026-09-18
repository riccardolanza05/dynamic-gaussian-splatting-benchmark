import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
R=runs.copy(); IT=R[R["mode"]=="iterations"].set_index(["method_short","scene"])
LO=R[R["mode"]=="target_eval_loss"].set_index(["method_short","scene"])
FAIR=["bouncingballs","hellwarrior","hook","mutant","standup"]
def v(t,m,s,c):
    try: return float(t.loc[(m,s),c])
    except Exception: return np.nan

# 36: the iteration/time inversion at equal L1 target
fig, axes = plt.subplots(1,4, figsize=(13.6,4.0))
items=[("Iterations to target", lambda m:[v(LO,m,s,"fc_total_iterations") for s in FAIR], "{:.0f}"),
       ("Training images seen", lambda m:[v(LO,m,s,"fc_images_seen") for s in FAIR], "{:.0f}"),
       ("Training time to target (min)", lambda m:[v(LO,m,s,"fc_training_time_s")/60 for s in FAIR], "{:.1f}"),
       ("Model storage at target (MB)", lambda m:[v(LO,m,s,"model_storage_mb") for s in FAIR], "{:.1f}")]
for ax,(tit,f,fmt) in zip(axes,items):
    for i,m in enumerate(ORDER):
        vs=f(m); mu=np.nanmean(vs)
        ax.bar(i, mu, 0.55, color=C[m], edgecolor=SURF, lw=1.2, zorder=3)
        ax.scatter([i]*len(vs), vs, s=26, color=SURF, edgecolor=C[m], lw=1.4, zorder=4)
        ax.text(i, mu, fmt.format(mu), ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels(ORDER, rotation=18, ha="right", fontsize=8, color=INK2)
    ax.set_title(tit+"  (lower is better)", color=INK, loc="left", fontweight="bold", fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.9); ax.set_axisbelow(True); ax.margins(y=0.18)
fig.suptitle("At equal reconstruction quality (same L1 target): fewer iterations does not mean less compute",
             color=INK, fontsize=12.5, fontweight="bold", x=0.012, ha="left", y=1.04)
fig.tight_layout()
savefig(fig,"36_iso_loss_cost_inversion",
  "Mean over the five scenes completed by all three methods; hollow markers are the individual scenes. "
  "4DGS native-4D needs the fewest optimisation steps only because it uses much larger per-scene batches: in training samples it consumes about three times more than the other two, and by far the most wall-clock time and disk space. "+CAVEAT)

# 37: PSNR at target vs storage (compactness pareto)
fig, ax = plt.subplots(figsize=(7.6,5.0))
for m in ORDER:
    for s in SCENES:
        x=v(LO,m,s,"model_storage_mb"); y=v(LO,m,s,"fc_psnr")
        if np.isnan(x) or np.isnan(y): continue
        ax.scatter(x,y,s=70,color=C[m],alpha=0.85,edgecolor=SURF,lw=1.2,zorder=3)
        ax.annotate(SCENE_LABEL[s],(x,y),fontsize=6.5,color=MUTED,xytext=(5,3),textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("Model storage at target (MB, log scale)"); ax.set_ylabel("PSNR at target (dB)")
ax.grid(True,alpha=0.9); ax.set_axisbelow(True)
fig.suptitle("Same reconstruction quality, very different model sizes", color=INK, fontsize=12.5,
             fontweight="bold", x=0.012, ha="left", y=1.02)
method_legend(fig, ncol=3, y=0.965)
fig.tight_layout(); savefig(fig,"37_storage_vs_psnr_at_target", CAVEAT)
