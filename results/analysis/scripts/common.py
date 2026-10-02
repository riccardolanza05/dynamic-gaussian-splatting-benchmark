import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.environ.get("GS_ROOT") or os.path.dirname(os.path.dirname(
           os.path.dirname(os.path.abspath(__file__))))
from study import ANALYSIS_DIR
OUT  = os.path.join(ROOT, ANALYSIS_DIR)
FIG  = os.path.join(OUT, "figures")
TAB  = os.path.join(OUT, "tables")
for d in (FIG, TAB): os.makedirs(d, exist_ok=True)

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; AXIS="#c3c2b7"

# Method and scene naming, colours, flags and caveats: one table per study
# (GS_STUDY=monocular by default, GS_STUDY=n3dv for the multi-view one).
from study import (STUDY, DATASET, C, ORDER, LABEL, SCENES, SCENE_LABEL,
                   FAIR, PARTIAL, NOT_EVAL, CAVEAT, FAIRNOTE, RADARNOTE,
                   MISSING_SCENE_FILL, HIGHLIGHT_SCENES, IMG_BUDGET,
                   NMETHODS_WORD, MEANNOTE, NOTE36, VARIANT_GROUPS, GPU, PA_BUDGET,
                   PA_BUDGET_CAP, PA_RUNS, PA_AT, PA_TOTAL, STEP_BUDGET, TITLE01, TITLE27,
                   NOTE39, SAMPLE_NOTE, TIME_NOTE, BSNOTE, NOTE43, GRID_NOTE, NOTE33)

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
if MISSING_SCENE_FILL:
    runs["scene"] = runs["scene"].fillna(MISSING_SCENE_FILL)
# total iterations including 4DGaussians coarse stage
curves["iter_total"] = curves["total_iterations"].fillna(curves["iteration"])
# PARTIAL and NOT_EVAL come from study.py; every figure honours them automatically.

# Variants of one method (Spacetime Gaussians lite / full) are siblings: one that was not
# run at all is dropped from the figures, so a study with a single variant shows only it.
# With no run of any sibling the first one is kept, as the placeholder of the method.
# ORDER is edited in place, so every script that imported it sees the same list.
_ran = set(runs.method_short.dropna()) if "method_short" in runs.columns else set()
for _group in VARIANT_GROUPS:
    _have = [m for m in _group if m in _ran] or _group[:1]
    for _m in _group:
        if _m not in _have and _m in ORDER:
            ORDER.remove(_m)

# Width of one bar in the grouped-bar figures: 0.26 for up to three methods (the layout
# the monocular figures were drawn with), narrower when there are more.
BAR_W = 0.26 if len(ORDER) <= 3 else 0.8 / len(ORDER)

# FAIR is the set of scenes the aggregates average over, and it is a claim: every method
# completed every scene in it. A half-finished loop would otherwise average over scenes a
# method never ran, silently. Warn rather than fail, so a partial run can still be inspected.
_missing = [(m, s) for m in ORDER for s in FAIR
            if runs[(runs.method_short == m) & (runs.scene == s)
                    & (runs["mode"] == "iterations")].empty]
if _missing:
    print("WARNING: FAIR lists %d method/scene pairs with no Protocol A run: %s"
          % (len(_missing), _missing))
    print("         The aggregate tables and figures will average over fewer runs than they")
    print("         claim. Trim FAIR in study.py, or finish the loop, before quoting them.")


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
