import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

R = runs.copy()
IT = R[R["mode"]=="iterations"].set_index(["method_short","scene"])
LO = R[R["mode"]=="target_eval_loss"].set_index(["method_short","scene"])
FAIR = ["bouncingballs","hellwarrior","hook","mutant","standup"]   # complete for all 3 methods
BLUE = LinearSegmentedColormap.from_list("b",["#cde2fb","#9ec5f4","#5598e7","#2a78d6","#184f95","#0d366b"])
DIV  = LinearSegmentedColormap.from_list("d",["#d03b3b","#f0efec","#2a78d6"])

def val(tab,m,s,c):
    try: return float(tab.loc[(m,s),c])
    except Exception: return np.nan

# ---- 25: heatmap panel of the four metrics (iso-iterations) ----
specs = [("best_psnr","Best PSNR (dB)","{:.2f}",False),("best_ssim","Best SSIM","{:.4f}",False),
         ("best_lpips","Best LPIPS","{:.4f}",True),("best_eval_l1_loss","Best eval L1","{:.4f}",True)]
fig, axes = plt.subplots(2,2, figsize=(13.2,6.0))
for ax,(col,tit,fmt,low) in zip(axes.ravel(), specs):
    M = np.array([[val(IT,m,s,col) for s in SCENES] for m in ORDER])
    Z = np.full_like(M, np.nan)
    for j in range(M.shape[1]):
        c = M[:,j]
        if np.all(np.isnan(c)): continue
        lo,hi = np.nanmin(c), np.nanmax(c)
        Z[:,j] = 0.5 if hi==lo else ((hi-c)/(hi-lo) if low else (c-lo)/(hi-lo))
    ax.imshow(Z, cmap=BLUE, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i,j]
            txt = "n/a" if np.isnan(v) else fmt.format(v)
            colr = MUTED if np.isnan(v) else ("#ffffff" if Z[i,j]>0.55 else INK)
            wt = "bold" if (not np.isnan(v) and Z[i,j]==1.0) else "normal"
            ax.text(j,i,txt,ha="center",va="center",fontsize=7.6,color=colr,fontweight=wt)
    ax.set_xticks(range(len(SCENES))); ax.set_xticklabels([SCENE_LABEL[s] for s in SCENES],
                  rotation=25, ha="right", fontsize=7.5, color=INK2)
    ax.set_yticks(range(3)); ax.set_yticklabels(ORDER, fontsize=8, color=INK2)
    ax.set_title(tit+("  (lower is better)" if low else ""), color=INK, loc="left", fontweight="bold", fontsize=9.5)
    ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
fig.suptitle("Per-scene comparison at an equal iteration budget — colour = rank within the scene (dark = best)",
             color=INK, fontsize=12.5, fontweight="bold", x=0.012, ha="left", y=1.03)
fig.tight_layout(); savefig(fig,"25_heatmap_iso_iterations", CAVEAT)

# ---- 26: PSNR gap to the best method, per scene ----
fig, ax = plt.subplots(figsize=(12.4,3.2))
M = np.array([[val(IT,m,s,"best_psnr") for s in SCENES] for m in ORDER])
D = M - np.nanmax(M,axis=0, keepdims=True)
im = ax.imshow(D, cmap=DIV, norm=TwoSlopeNorm(vmin=np.nanmin(D), vcenter=0, vmax=0.001), aspect="auto")
for i in range(3):
    for j in range(len(SCENES)):
        v=D[i,j]
        ax.text(j,i,"n/a" if np.isnan(v) else ("best" if v==0 else f"{v:+.2f}"),
                ha="center",va="center",fontsize=8,color=INK if (np.isnan(v) or v>-4) else "#ffffff",
                fontweight="bold" if v==0 else "normal")
ax.set_xticks(range(len(SCENES))); ax.set_xticklabels([SCENE_LABEL[s] for s in SCENES], fontsize=8.5, color=INK2)
ax.set_yticks(range(3)); ax.set_yticklabels(ORDER, fontsize=9, color=INK2); ax.tick_params(length=0)
[sp.set_visible(False) for sp in ax.spines.values()]
fig.suptitle("PSNR gap to the best method on each scene (dB) — equal-iteration budget",
             color=INK, fontsize=12.5, fontweight="bold", x=0.012, ha="left", y=1.06)
fig.tight_layout(); savefig(fig,"26_psnr_gap_to_best", CAVEAT)

# ---- 27: peak & degradation analysis ----
fig, axes = plt.subplots(1,3, figsize=(13.2,4.1), sharey=True)
for ax,m in zip(axes,ORDER):
    for s in SCENES:
        d = cur(m,s,"iterations")
        if d.empty or d.psnr.isna().all(): continue
        pk = d.psnr.max()
        ax.plot(d.iter_total, d.psnr-pk, color=C[m], lw=1.5, alpha=0.55,
                ls="--" if is_partial(m,s,"iterations") else "-")
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_title(LABEL[m], color=INK, loc="left", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.9); ax.set_axisbelow(True); ax.set_ylim(-6,1)
    ax.set_xlabel("Training iterations", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v,p: f"{v/1000:g}k" if v>=1000 else f"{v:g}")
axes[0].set_ylabel("PSNR − peak PSNR of the run (dB)")
fig.suptitle("Post-peak behaviour: 4DGS native-4D peaks within the first thousands of iterations and then degrades",
             color=INK, fontsize=12.5, fontweight="bold", x=0.012, ha="left", y=1.02)
fig.tight_layout(); savefig(fig,"27_peak_and_degradation", CAVEAT+" One line per scene.")

# ---- 28: peak iteration vs drop at 30k ----
fig, axes = plt.subplots(1,2, figsize=(12.6,4.0))
x = np.arange(len(SCENES)); w=0.26
for i,m in enumerate(ORDER):
    pk  = [val(IT,m,s,"best_psnr_iter") for s in SCENES]
    drp = [val(IT,m,s,"best_psnr")-val(IT,m,s,"final_psnr") for s in SCENES]
    axes[0].bar(x+(i-1)*w, pk, w*0.92, color=C[m], edgecolor=SURF, lw=1.2,
                hatch=["///" if is_partial(m,s,"iterations") else "" for s in SCENES], zorder=3)
    axes[1].bar(x+(i-1)*w, drp, w*0.92, color=C[m], edgecolor=SURF, lw=1.2,
                hatch=["///" if is_partial(m,s,"iterations") else "" for s in SCENES], zorder=3)
for ax,t,yl in zip(axes,["Iteration of peak PSNR","PSNR lost between peak and end of run"],
                   ["Iteration","Peak PSNR − final PSNR (dB)"]):
    ax.set_xticks(x); ax.set_xticklabels([SCENE_LABEL[s] for s in SCENES], rotation=25, ha="right", fontsize=8, color=INK2)
    ax.set_ylabel(yl); ax.grid(True, axis="y", alpha=0.9); ax.set_axisbelow(True)
    ax.set_title(t, color=INK, loc="left", fontweight="bold", fontsize=10)
method_legend(fig, extra=[Patch(facecolor=SURF, edgecolor=MUTED, hatch="///", label="partial run")], y=1.06)
fig.tight_layout(); savefig(fig,"28_peak_iteration_and_drop", CAVEAT)

# ---- 29: iterations / time to reach PSNR thresholds ----
def first_reach(m,s,th,mode="iterations",col="iter_total"):
    d = cur(m,s,mode)
    if d.empty: return np.nan
    h = d[d.psnr>=th]
    return np.nan if h.empty else float(h.iloc[0][col])
fig, axes = plt.subplots(1,2, figsize=(12.8,4.0))
for k,(th,ax) in enumerate(zip([25,30],axes)):
    for i,m in enumerate(ORDER):
        v=[first_reach(m,s,th) for s in SCENES]
        ax.bar(x+(i-1)*w, v, w*0.92, color=C[m], edgecolor=SURF, lw=1.2, zorder=3)
        for xi,vv,s in zip(x+(i-1)*w, v, SCENES):
            if np.isnan(vv):
                ax.text(xi, 200, "never" if not cur(m,s,"iterations").empty else "n/a",
                        rotation=90, fontsize=7, color=MUTED, ha="center", va="bottom", style="italic")
    ax.set_xticks(x); ax.set_xticklabels([SCENE_LABEL[s] for s in SCENES], rotation=25, ha="right", fontsize=8, color=INK2)
    ax.set_ylabel("Iterations"); ax.grid(True, axis="y", alpha=0.9); ax.set_axisbelow(True)
    ax.set_title(f"Iterations to first reach PSNR ≥ {th} dB", color=INK, loc="left", fontweight="bold", fontsize=10)
method_legend(fig, y=1.06)
fig.tight_layout(); savefig(fig,"29_iterations_to_psnr_threshold", CAVEAT+" Evaluation grid: every 1000 iterations.")

# ---- 30: quality vs cost scatter (iso-iterations) ----
fig, axes = plt.subplots(1,2, figsize=(13.0,4.6))
for m in ORDER:
    for s in SCENES:
        t=val(IT,m,s,"final_training_time_s"); q=val(IT,m,s,"best_psnr")
        vram=val(IT,m,s,"peak_vram_mb")
        if np.isnan(t) or np.isnan(q): continue
        axes[0].scatter(t/60, q, s=(0 if np.isnan(vram) else vram)/12+30, color=C[m],
                        alpha=0.85, edgecolor=SURF, linewidth=1.2, zorder=3)
        axes[0].annotate(SCENE_LABEL[s], (t/60,q), fontsize=6.5, color=MUTED,
                         xytext=(4,4), textcoords="offset points")
for m in ORDER:
    for s in SCENES:
        t=val(LO,m,s,"fc_training_time_s"); q=val(LO,m,s,"fc_psnr")
        if np.isnan(t) or np.isnan(q): continue
        axes[1].scatter(t/60, q, s=60, color=C[m], alpha=0.85, edgecolor=SURF, linewidth=1.2, zorder=3)
        axes[1].annotate(SCENE_LABEL[s], (t/60,q), fontsize=6.5, color=MUTED,
                         xytext=(4,4), textcoords="offset points")
axes[0].set_title("Equal-iteration budget — marker size ∝ peak VRAM", color=INK, loc="left", fontweight="bold", fontsize=10)
axes[1].set_title("Equal-L1-target budget", color=INK, loc="left", fontweight="bold", fontsize=10)
for ax in axes:
    ax.set_xlabel("Training time (minutes, Tesla T4)"); ax.set_ylabel("Best PSNR (dB)")
    ax.grid(True, alpha=0.9); ax.set_axisbelow(True)
method_legend(fig, y=1.06)
fig.suptitle("Quality against compute cost", color=INK, fontsize=12.5, fontweight="bold", x=0.012, ha="left", y=1.14)
fig.tight_layout(); savefig(fig,"30_quality_vs_cost_scatter", CAVEAT)
