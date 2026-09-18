import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.environ.get("GS_ROOT") or os.path.dirname(os.path.dirname(
           os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(ROOT, "analysis")
FIG  = os.path.join(OUT, "figures")
TAB  = os.path.join(OUT, "tables")
for d in (FIG, TAB): os.makedirs(d, exist_ok=True)

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; AXIS="#c3c2b7"
C = {"4DGaussians":"#2a78d6", "4DGS-fudan":"#eb6834", "Deformable-3DGS":"#1baf7a"}
ORDER = ["4DGaussians","4DGS-fudan","Deformable-3DGS"]
LABEL = {"4DGaussians":"4DGaussians (Wu et al.)",
         "4DGS-fudan":"4DGS native-4D (fudan-zvg)",
         "Deformable-3DGS":"Deformable-3DGS (Yang et al.)"}
SCENES = ["bouncingballs","hellwarrior","hook","jumpingjacks","lego","mutant","standup","trex"]
SCENE_LABEL = {"bouncingballs":"Bouncing Balls","hellwarrior":"Hell Warrior","hook":"Hook",
               "jumpingjacks":"Jumping Jacks","lego":"Lego","mutant":"Mutant",
               "standup":"Stand Up","trex":"T-Rex"}

plt.rcParams.update({
 "figure.facecolor":SURF, "axes.facecolor":SURF, "savefig.facecolor":SURF,
 "font.family":"sans-serif",
 "font.sans-serif":["DejaVu Sans"],
 "font.size":9, "axes.titlesize":10, "axes.labelsize":9,
 "axes.edgecolor":AXIS, "axes.linewidth":0.8, "axes.labelcolor":INK2,
 "axes.spines.top":False, "axes.spines.right":False,
 "xtick.color":MUTED, "ytick.color":MUTED, "xtick.labelsize":8, "ytick.labelsize":8,
 "grid.color":GRID, "grid.linewidth":0.6, "legend.frameon":False, "legend.fontsize":8.5,
 "lines.linewidth":2.0, "figure.dpi":110,
})

curves = pd.read_csv(os.path.join(TAB,"curves_all.csv"))
runs   = pd.read_csv(os.path.join(TAB,"runs_summary.csv"))
runs["scene"] = runs["scene"].fillna("lego")
# total iterations including 4DGaussians coarse stage
curves["iter_total"] = curves["total_iterations"].fillna(curves["iteration"])
# flags
PARTIAL = {("4DGS-fudan","trex","iterations"), ("4DGS-fudan","jumpingjacks","iterations")}
NOT_EVAL = {("4DGS-fudan","lego")}

def is_partial(m,s,mode): return (m,s,mode) in PARTIAL
def cur(m,s,mode):
    d = curves[(curves.method_short==m)&(curves.scene==s)&(curves["mode"]==mode)]
    return d.sort_values("iter_total")

import textwrap as _tw
def savefig(fig, name, note=None):
    if note:
        w = max(90, int(fig.get_size_inches()[0]*16))
        fig.text(0.005, 0.005, "\n".join(_tw.wrap(note, w)), fontsize=7, color=MUTED,
                 ha="left", va="top")
    fig.savefig(os.path.join(FIG,name+".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG,name+".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

def method_legend(fig, extra=None, ncol=4, y=0.995):
    h=[Line2D([],[],color=C[m],lw=2.4,label=LABEL[m]) for m in ORDER]
    if extra: h += extra
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5,y), ncol=ncol)

CAVEAT = ("Caveats: Lego could not be evaluated for 4DGS native-4D (known limitation reported in the original paper). "
          "T-Rex and Jumping Jacks fixed-iteration runs of 4DGS native-4D are partial (Colab T4 budget) and are marked as such.")
