import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from matplotlib.colors import LinearSegmentedColormap
R = runs.copy()
IT = R[R["mode"]=="iterations"].set_index(["method_short","scene"])
LO = R[R["mode"]=="target_eval_loss"].set_index(["method_short","scene"])
FAIR = ["bouncingballs","hellwarrior","hook","mutant","standup"]
def val(tab,m,s,c):
    try: return float(tab.loc[(m,s),c])
    except Exception: return np.nan
FAIRNOTE = ("Aggregates are computed on the five scenes completed by all three methods "
            "(Bouncing Balls, Hell Warrior, Hook, Mutant, Stand Up); Lego, T-Rex and Jumping Jacks are excluded. ")

# ---- 31: aggregate dot plot ----
panels = [("best_psnr","Best PSNR (dB)",IT,False),("best_ssim","Best SSIM",IT,False),
          ("best_lpips","Best LPIPS",IT,True),("best_eval_l1_loss","Best eval L1",IT,True)]
fig, axes = plt.subplots(1,4, figsize=(13.4,4.0))
for ax,(col,tit,tab,low) in zip(axes,panels):
    for i,m in enumerate(ORDER):
        vs=[val(tab,m,s,col) for s in FAIR]
        ax.scatter([i]*len(vs), vs, s=34, color=C[m], alpha=0.45, edgecolor=SURF, lw=1, zorder=3)
        mu=np.nanmean(vs)
        ax.plot([i-0.26,i+0.26],[mu,mu], color=C[m], lw=3, zorder=4)
        ax.text(i+0.30, mu, f"{mu:.4g}", fontsize=8, color=INK2, va="center")
    ax.set_xticks(range(3)); ax.set_xticklabels(ORDER, rotation=18, ha="right", fontsize=8, color=INK2)
    ax.set_title(tit+("  ↓ better" if low else "  ↑ better"), color=INK, loc="left", fontweight="bold", fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.9); ax.set_axisbelow(True); ax.set_xlim(-0.6,2.8)
fig.suptitle("Aggregate quality at an equal iteration budget — dots are scenes, bar is the mean",
             color=INK, fontsize=12.5, fontweight="bold", x=0.012, ha="left", y=1.03)
fig.tight_layout(); savefig(fig,"31_aggregate_quality_dots", FAIRNOTE+CAVEAT)

# ---- 32: radar ----
axes_def = [("PSNR",      lambda m: np.nanmean([val(IT,m,s,"best_psnr") for s in FAIR]), False),
            ("SSIM",      lambda m: np.nanmean([val(IT,m,s,"best_ssim") for s in FAIR]), False),
            ("LPIPS",     lambda m: np.nanmean([val(IT,m,s,"best_lpips") for s in FAIR]), True),
            ("Training\nspeed", lambda m: np.nanmean([val(IT,m,s,"final_training_time_s") for s in FAIR]), True),
            ("GPU memory",lambda m: np.nanmean([val(IT,m,s,"peak_vram_mb") for s in FAIR]), True),
            ("Model\ncompactness", lambda m: np.nanmean([val(LO,m,s,"model_storage_mb") for s in FAIR]), True),
            ("Training samples\nto L1 target", lambda m: np.nanmean([val(LO,m,s,"fc_images_seen") for s in FAIR]), True)]
raw = {m:[f(m) for _,f,_ in axes_def] for m in ORDER}
N=len(axes_def)
norm={m:[] for m in ORDER}
for k in range(N):
    col=np.array([raw[m][k] for m in ORDER]); lo,hi=np.nanmin(col),np.nanmax(col)
    for m in ORDER:
        v=raw[m][k]
        z=0.5 if hi==lo else ((hi-v)/(hi-lo) if axes_def[k][2] else (v-lo)/(hi-lo))
        norm[m].append(0.12+0.88*z)
ang=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); ang+=ang[:1]
fig=plt.figure(figsize=(7.6,6.4)); ax=fig.add_subplot(111,polar=True)
ax.set_facecolor(SURF)
for m in ORDER:
    v=norm[m]+norm[m][:1]
    ax.plot(ang,v,color=C[m],lw=2.2); ax.fill(ang,v,color=C[m],alpha=0.10)
ax.set_xticks(ang[:-1]); ax.set_xticklabels([a[0] for a in axes_def], fontsize=8.5, color=INK2)
ax.set_yticks([0.25,0.5,0.75,1.0]); ax.set_yticklabels([]); ax.set_ylim(0,1.05)
ax.grid(color=GRID); ax.spines["polar"].set_color(AXIS)
ax.set_position([0.13,0.16,0.74,0.72])
fig.suptitle("Normalised profile of the three methods (outer = better)", color=INK,
             fontsize=12.5, fontweight="bold", y=1.02)
method_legend(fig, ncol=1, y=0.115)
savefig(fig,"32_radar_profile", FAIRNOTE+"Each axis is min–max normalised across the three methods; it shows ranking, not absolute values.")

# ---- 33: per-method detail sheets ----
RAMP = LinearSegmentedColormap.from_list("s",["#9ec5f4","#0d366b"])
for m in ORDER:
    fig, axes = plt.subplots(2,2, figsize=(12.4,6.6))
    specs=[("psnr","PSNR (dB)",False),("lpips","LPIPS (VGG)",True),
           ("num_gaussians","# Gaussians",True),("training_time_s","Training time (s)",False)]
    scs=[s for s in SCENES if not cur(m,s,"iterations").empty]
    cols={s:RAMP(i/max(len(scs)-1,1)) for i,s in enumerate(scs)}
    for ax,(met,yl,logy) in zip(axes.ravel(),specs):
        for s in scs:
            d=cur(m,s,"iterations")
            ax.plot(d.iter_total,d[met],color=cols[s],lw=1.8,
                    ls="--" if is_partial(m,s,"iterations") else "-")
            ax.annotate(SCENE_LABEL[s],(d.iter_total.iloc[-1],d[met].iloc[-1]),fontsize=6.5,
                        color=cols[s],xytext=(3,0),textcoords="offset points",va="center")
        if logy: ax.set_yscale("log")
        ax.set_ylabel(yl); ax.set_xlabel("Training iterations", fontsize=8)
        ax.grid(True,alpha=0.9); ax.set_axisbelow(True); ax.margins(x=0.16)
        ax.xaxis.set_major_formatter(lambda v,p: f"{v/1000:g}k" if v>=1000 else f"{v:g}")
    fig.suptitle(f"{LABEL[m]} — per-scene training behaviour (equal-iteration runs)",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.012, ha="left", y=1.02)
    fig.tight_layout()
    savefig(fig, f"33_{m.lower().replace(' ','').replace('-','_')}_detail",
            "Dashed = partial run. "+ (("Lego is not evaluable for this method (known limitation of the original paper). ") if m=="4DGS-fudan" else ""))

# ---- 34: efficiency ratios ----
x=np.arange(len(SCENES)); w=0.26
fig, axes = plt.subplots(1,2, figsize=(13.0,4.2))
for i,m in enumerate(ORDER):
    ppm=[val(IT,m,s,"best_psnr")/(val(IT,m,s,"final_training_time_s")/60) for s in SCENES]
    ppmb=[val(LO,m,s,"fc_psnr")/val(LO,m,s,"model_storage_mb") for s in SCENES]
    axes[0].bar(x+(i-1)*w,ppm,w*0.92,color=C[m],edgecolor=SURF,lw=1.2,zorder=3)
    axes[1].bar(x+(i-1)*w,ppmb,w*0.92,color=C[m],edgecolor=SURF,lw=1.2,zorder=3)
for ax,t,yl in zip(axes,["Quality per minute of training (equal-iteration runs)",
                         "Quality per MB of stored model (equal-L1-target runs)"],
                   ["Best PSNR / training minute","PSNR at target / model MB"]):
    ax.set_xticks(x); ax.set_xticklabels([SCENE_LABEL[s] for s in SCENES],rotation=25,ha="right",fontsize=8,color=INK2)
    ax.set_ylabel(yl); ax.grid(True,axis="y",alpha=0.9); ax.set_axisbelow(True)
    ax.set_title(t,color=INK,loc="left",fontweight="bold",fontsize=10)
method_legend(fig,y=1.06); fig.tight_layout(); savefig(fig,"34_efficiency_ratios",CAVEAT)

# ---- 35: speed-up to reach the same L1 target ----
fig, ax = plt.subplots(figsize=(12.4,4.0))
base=[np.nanmax([val(LO,m,s,"fc_training_time_s") for m in ORDER]) for s in SCENES]
for i,m in enumerate(ORDER):
    sp=[base[j]/val(LO,m,s,"fc_training_time_s") for j,s in enumerate(SCENES)]
    b=ax.bar(x+(i-1)*w,sp,w*0.92,color=C[m],edgecolor=SURF,lw=1.2,zorder=3)
    for xi,v in zip(x+(i-1)*w,sp):
        if not np.isnan(v): ax.text(xi,v,f"{v:.2f}×",ha="center",va="bottom",fontsize=7,color=INK2)
ax.axhline(1,color=MUTED,lw=0.8,ls=":")
ax.set_xticks(x); ax.set_xticklabels([SCENE_LABEL[s] for s in SCENES],fontsize=8.5,color=INK2)
ax.set_ylabel("Speed-up vs the slowest method on that scene"); ax.grid(True,axis="y",alpha=0.9); ax.set_axisbelow(True)
fig.suptitle("How much faster each method reaches the same L1 target",color=INK,fontsize=12.5,
             fontweight="bold",x=0.012,ha="left",y=1.10)
method_legend(fig,y=1.03); fig.tight_layout(); savefig(fig,"35_speedup_to_target",CAVEAT)
