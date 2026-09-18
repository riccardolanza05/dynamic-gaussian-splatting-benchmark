import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
R=runs[runs["mode"]=="iterations"].set_index(["method_short","scene"])
def bs(m,s):
    try: return int(R.loc[(m,s),"batch_size"])
    except Exception: return None
def at(m,s,xcol,xval,ycol):
    """metric of method m on scene s when the budget xcol reaches xval (linear interp)."""
    d=cur(m,s,"iterations")
    if d.empty: return np.nan
    x=d[xcol].to_numpy(float); y=d[ycol].to_numpy(float)
    ok=~np.isnan(x)&~np.isnan(y); x,y=x[ok],y[ok]
    if len(x)<2 or xval>x.max(): return np.nan
    return float(np.interp(xval,x,y))

IMG_BUDGET=30000
def tbudget(s):
    v=[cur(m,s,"iterations").training_time_s.max() for m in ORDER if not cur(m,s,"iterations").empty]
    return float(np.nanmin(v))

# ---- 39: the same runs on three budget axes ----
rows=["hellwarrior","bouncingballs","mutant","trex"]
axes_def=[("iter_total","Optimisation steps",lambda v: f"{v/1000:g}k" if v>=1000 else f"{v:g}"),
          ("images_seen","Training images seen",lambda v: f"{v/1000:g}k" if v>=1000 else f"{v:g}"),
          ("training_time_s","Wall-clock training time (s)",lambda v: f"{v/60:g}m" if v>=60 else f"{v:g}s")]
fig,axg=plt.subplots(len(rows),3,figsize=(13.2,11.2))
for i,s in enumerate(rows):
    for j,(xc,xl,f) in enumerate(axes_def):
        ax=axg[i,j]
        for m in ORDER:
            d=cur(m,s,"iterations")
            if d.empty or d.psnr.isna().all(): continue
            ax.plot(d[xc],d.psnr,color=C[m],lw=2.0,ls="--" if is_partial(m,s,"iterations") else "-")
        if j==1: ax.axvline(IMG_BUDGET,color=MUTED,lw=1.0,ls=":")
        if j==2: ax.axvline(tbudget(s),color=MUTED,lw=1.0,ls=":")
        ax.grid(True,alpha=.9); ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(lambda v,p,f=f: f(v))
        if j==1: ax.set_xscale("log"); ax.set_xlim(left=200)
        if j==2: ax.set_xscale("log"); ax.set_xlim(left=2)
        if i==0: ax.set_title(xl,color=INK,loc="left",fontweight="bold",fontsize=10)
        if i==len(rows)-1: ax.set_xlabel(xl,fontsize=8)
        if j==0:
            ax.set_ylabel("PSNR (dB)")
            ax.text(-0.28,0.5,f"{SCENE_LABEL[s]}\n4DGS-fudan batch = {bs('4DGS-fudan',s)}",
                    transform=ax.transAxes,rotation=90,va="center",ha="center",
                    fontsize=9,color=INK,fontweight="bold")
fig.suptitle("The same runs, three budget axes — the ranking depends on what you call a budget",
             color=INK,fontsize=12.5,fontweight="bold",x=0.012,ha="left",y=1.015)
method_legend(fig,extra=[Line2D([],[],color=MUTED,lw=1,ls=":",label="common budget used in figures 40–41")],y=0.995)
fig.tight_layout()
savefig(fig,"39_three_budget_axes",
 "Rows are ordered by the per-scene batch size of 4DGS-fudan (1, 2, 8, 24); the other two methods always use batch 1. "
 "Columns 2 and 3 are on a log axis because the sample and time budgets span an order of magnitude between methods. "+CAVEAT)

# ---- 40: equal number of training samples ----
def grouped(getter,ylabel,title,name,fmt="{:.2f}",lower=False,note="",annotate=True,logy=False):
    x=np.arange(len(SCENES)); w=0.26
    fig,ax=plt.subplots(figsize=(12.6,4.2)); nan=[]
    for i,m in enumerate(ORDER):
        for xi,s in zip(x+(i-1)*w,SCENES):
            v=getter(m,s)
            if v is None or (isinstance(v,float) and np.isnan(v)): nan.append(xi); continue
            ax.bar(xi,v,w*0.92,color=C[m],edgecolor=SURF,lw=1.2,zorder=3)
            if annotate: ax.text(xi,v,fmt.format(v),ha="center",va="bottom",fontsize=6.8,color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([SCENE_LABEL[s] for s in SCENES],fontsize=8.5,color=INK2)
    ax.set_ylabel(ylabel); ax.grid(True,axis="y",alpha=.9); ax.set_axisbelow(True); ax.margins(y=.16)
    if logy: ax.set_yscale("log")
    y0,y1=ax.get_ylim()
    for xi in nan: ax.text(xi,y0+0.02*(y1-y0),"n/a",ha="center",va="bottom",fontsize=8,color=MUTED,rotation=90,style="italic")
    fig.suptitle(title+("  (lower is better)" if lower else "  (higher is better)"),
                 color=INK,fontsize=12.5,fontweight="bold",x=0.012,ha="left",y=1.115)
    method_legend(fig,y=1.045); fig.tight_layout(); savefig(fig,name,note)

SAMPLE_NOTE=("Every method is stopped after the same number of training images (30 000), which is the whole "
             "Protocol-A budget of the batch-1 methods. Under this budget the partial 4DGS-fudan runs are complete: "
             "at 30 000 samples T-Rex has run 1 250 of its 6 000 recorded steps and Jumping Jacks 1 875 of 15 000. "
             "Lego stays unavailable for 4DGS-fudan. Values are linearly interpolated on the 1 000-iteration evaluation grid.")
grouped(lambda m,s: at(m,s,"images_seen",IMG_BUDGET,"psnr"),"PSNR (dB)",
        "Equal number of training samples — PSNR after 30 000 training images","40_iso_samples_psnr",
        note=SAMPLE_NOTE)
grouped(lambda m,s: at(m,s,"images_seen",IMG_BUDGET,"iter_total"),"Optimisation steps used",
        "Optimisation steps consumed by the same 30 000 training images","41_iso_samples_steps",
        fmt="{:.0f}",lower=True,note=SAMPLE_NOTE)

# ---- 42: equal wall-clock ----
TIME_NOTE=("Each scene is compared at the same wall-clock budget: the time the fastest method needed to finish its "
           "30 000 steps on that scene (756 s to 2 284 s, always set by 4DGaussians). This is the budget a fixed "
           "Colab T4 session actually buys. "+CAVEAT)
grouped(lambda m,s: at(m,s,"training_time_s",tbudget(s),"psnr"),"PSNR (dB)",
        "Equal wall-clock budget — PSNR when the fastest method finishes its run","42_iso_time_psnr",
        note=TIME_NOTE)

# ---- 43: throughput per 1000 samples, not per 1000 steps ----
fig,axes=plt.subplots(1,2,figsize=(13.0,4.2)); x=np.arange(len(SCENES)); w=0.26
for i,m in enumerate(ORDER):
    per_it,per_img=[],[]
    for s in SCENES:
        d=cur(m,s,"iterations")
        if d.empty: per_it.append(np.nan); per_img.append(np.nan); continue
        t=d.training_time_s.max()
        per_it.append(1000*t/d.iter_total.max()); per_img.append(1000*t/d.images_seen.max())
    axes[0].bar(x+(i-1)*w,per_it,w*.92,color=C[m],edgecolor=SURF,lw=1.2,zorder=3)
    axes[1].bar(x+(i-1)*w,per_img,w*.92,color=C[m],edgecolor=SURF,lw=1.2,zorder=3)
for ax,t,yl in zip(axes,["Seconds per 1 000 optimisation steps","Seconds per 1 000 training images"],
                   ["s / 1 000 steps","s / 1 000 images"]):
    ax.set_xticks(x); ax.set_xticklabels([SCENE_LABEL[s] for s in SCENES],rotation=25,ha="right",fontsize=8,color=INK2)
    ax.set_ylabel(yl); ax.grid(True,axis="y",alpha=.9); ax.set_axisbelow(True)
    ax.set_title(t+"  (lower is better)",color=INK,loc="left",fontweight="bold",fontsize=10)
method_legend(fig,y=1.06); fig.tight_layout()
savefig(fig,"43_throughput_steps_vs_samples",
 "Left: cost of a step. Right: the same runs priced per training image — the batch-corrected throughput. "
 "The gap between the two panels is exactly the batch size of 4DGS-fudan on that scene. "+CAVEAT)

# ---- 44: who wins under which budget ----
from matplotlib.colors import ListedColormap
budgets=[("Equal steps\n(30 000 iterations)", lambda m,s: at(m,s,"iter_total",30000,"psnr")),
         ("Equal samples\n(30 000 images)",   lambda m,s: at(m,s,"images_seen",IMG_BUDGET,"psnr")),
         ("Equal wall-clock\n(fastest method's run)", lambda m,s: at(m,s,"training_time_s",tbudget(s),"psnr"))]
fig,ax=plt.subplots(figsize=(12.4,3.6))
Z=np.zeros((len(budgets),len(SCENES))); lab=[[""]*len(SCENES) for _ in budgets]
for i,(bn,f) in enumerate(budgets):
    for j,s in enumerate(SCENES):
        vals={m:f(m,s) for m in ORDER}
        good={m:v for m,v in vals.items() if v is not None and not np.isnan(v)}
        if not good: Z[i,j]=np.nan; lab[i][j]="—"; continue
        win=max(good,key=good.get); Z[i,j]=ORDER.index(win)
        srt=sorted(good.values(),reverse=True)
        gap=srt[0]-srt[1] if len(srt)>1 else np.nan
        lab[i][j]=f"{win}\n+{gap:.2f} dB" if not np.isnan(gap) else win
cmap=ListedColormap([C[m] for m in ORDER])
ax.imshow(np.ma.masked_invalid(Z),cmap=cmap,vmin=-0.5,vmax=2.5,aspect="auto",alpha=.88)
for i in range(len(budgets)):
    for j in range(len(SCENES)):
        ax.text(j,i,lab[i][j],ha="center",va="center",fontsize=7.4,color="#ffffff",fontweight="bold")
ax.set_xticks(range(len(SCENES))); ax.set_xticklabels([SCENE_LABEL[s] for s in SCENES],fontsize=8.5,color=INK2)
ax.set_yticks(range(len(budgets))); ax.set_yticklabels([b[0] for b in budgets],fontsize=8.5,color=INK2)
ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
fig.suptitle("Best PSNR under three definitions of the same budget — and by how much",
             color=INK,fontsize=12.5,fontweight="bold",x=0.012,ha="left",y=1.06)
fig.tight_layout()
savefig(fig,"44_winner_by_budget_definition",
 "Cell colour and text name the method with the highest PSNR at that budget; the number is its margin over the runner-up. "+CAVEAT)
