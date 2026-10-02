"""Which study the pipeline is analysing: the monocular one or a multi-view one.

The studies are reported separately (different datasets, different scenes, different
methods), so every table that names methods or scenes, and every sentence a figure prints
about the protocol, lives here and is selected by the GS_STUDY environment variable:

    GS_STUDY=monocular   (default)  D-NeRF, three methods       -> results/
    GS_STUDY=n3dv                   Neural 3D Video, all 300 frames (the main study),
                                    three methods               -> results/n3dv/analysis/
    GS_STUDY=n3dv_f50               Neural 3D Video, first 50 frames (the short-window
                                    study, with Protocol B)     -> results/n3dv/analysis_f50/

The two N3DV studies share one data root: their runs sit side by side in the same
method folders, told apart by the frame window every benchmark JSON records
(num_frames), and each study keeps only the runs of its own window (NUM_FRAMES).

The data root is still GS_ROOT, exactly as before; the two variables are set
together, e.g.

    GS_ROOT=../../n3dv GS_STUDY=n3dv ./run_all.sh

With GS_STUDY unset every value below is the one the scripts used before this
module existed, so the monocular outputs are unchanged.

Colours come from a colourblind-safe categorical palette. The two methods that
appear in both studies keep their hue across them, so a reader moving between the
monocular and the multi-view figures is not asked to relearn the legend.
"""

import os

STUDY = os.environ.get("GS_STUDY", "monocular").lower()
if STUDY not in ("monocular", "n3dv", "n3dv_f50"):
    raise SystemExit("GS_STUDY must be 'monocular', 'n3dv' or 'n3dv_f50', not %r" % STUDY)

# Where under GS_ROOT the tables, figures and dashboard go, and which frame window the
# runs must have been trained on (None: no filter, as in the monocular study).
ANALYSIS_DIR = "analysis"
NUM_FRAMES = None

# A method folder that holds several variants of one method, each reported as its own row:
# folder -> (the config key that names the variant, {value: long label}, default value).
# VARIANT_GROUPS lists the short tags of such siblings; a sibling with no run at all is
# left out of the figures, so a study that ran only one variant shows only that one.
VARIANTS = {}
VARIANT_GROUPS = []

BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#cc79a7"

if STUDY == "monocular":
    DATASET = "D-NeRF (monocular)"

    # folder under the data root -> long label
    METHOD_DIR = {
        "4dgaussian_output": "4DGaussians (Wu et al.)",
        "4dgs_fudan_output": "4DGS native-4D (fudan-zvg)",
        "deformablegaussian": "Deformable-3DGS (Yang et al.)",
    }
    SHORT = {
        "4DGaussians (Wu et al.)": "4DGaussians",
        "4DGS native-4D (fudan-zvg)": "4DGS-fudan",
        "Deformable-3DGS (Yang et al.)": "Deformable-3DGS",
    }
    C = {"4DGaussians": BLUE, "4DGS-fudan": ORANGE, "Deformable-3DGS": AQUA}
    ORDER = ["4DGaussians", "4DGS-fudan", "Deformable-3DGS"]
    LABEL = {
        "4DGaussians": "4DGaussians (Wu et al.)",
        "4DGS-fudan": "4DGS native-4D (fudan-zvg)",
        "Deformable-3DGS": "Deformable-3DGS (Yang et al.)",
    }
    SCENES = ["bouncingballs", "hellwarrior", "hook", "jumpingjacks",
              "lego", "mutant", "standup", "trex"]
    SCENE_LABEL = {"bouncingballs": "Bouncing Balls", "hellwarrior": "Hell Warrior",
                   "hook": "Hook", "jumpingjacks": "Jumping Jacks", "lego": "Lego",
                   "mutant": "Mutant", "standup": "Stand Up", "trex": "T-Rex"}
    # Scenes completed by every method: the aggregates are computed on these.
    FAIR = ["bouncingballs", "hellwarrior", "hook", "mutant", "standup"]
    # (method, scene, mode) triples whose run was stopped before its budget.
    PARTIAL = {("4DGS-fudan", "trex", "iterations"),
               ("4DGS-fudan", "jumpingjacks", "iterations")}
    # (method, scene) pairs that cannot be evaluated at all.
    NOT_EVAL = {("4DGS-fudan", "lego")}
    CAVEAT = ("Caveats: Lego could not be evaluated for 4DGS native-4D (known limitation "
              "reported in the original paper). T-Rex and Jumping Jacks fixed-iteration "
              "runs of 4DGS native-4D are partial (Colab T4 budget) and are marked as such.")
    FAIRNOTE = ("Aggregates are computed on the five scenes completed by all three methods "
                "(Bouncing Balls, Hell Warrior, Hook, Mutant, Stand Up); Lego, T-Rex and Jumping Jacks are excluded. ")
    RADARNOTE = "Each axis is min–max normalised across the three methods; it shows ranking, not absolute values."
    NMETHODS_WORD = "three"
    MEANNOTE = ("Mean over the five scenes completed by all three methods; "
                "hollow markers are the individual scenes. ")
    # Figure 36 states a finding of this study; a study with no runs yet states none.
    NOTE36 = ("4DGS native-4D needs the fewest optimisation steps only because it uses much "
              "larger per-scene batches: in training samples it consumes about three times "
              "more than the other two, and by far the most wall-clock time and disk space. ")
    # The fudan-zvg lego run writes no scene into its JSON; the column is filled in.
    MISSING_SCENE_FILL = "lego"
    # Scenes shown one per row in the three-budget-axes figure, and the iso-samples
    # budget (in training images) that every method reaches in its Protocol A run.
    HIGHLIGHT_SCENES = ["hellwarrior", "bouncingballs", "mutant", "trex"]
    IMG_BUDGET = 30000

    # --- What the figures say about the protocol and the machine -----------------------
    GPU = "Tesla T4"
    # Protocol A is "the same number of steps for everyone" in this study.
    PA_BUDGET = "equal-iteration budget"
    PA_BUDGET_CAP = "Equal-iteration budget"
    PA_RUNS = "equal-iteration runs"
    PA_AT = "at an equal iteration budget"
    PA_TOTAL = "(30k total iterations)"
    # The step count every method reaches in Protocol A (None: there is no common one).
    STEP_BUDGET = 30000
    TITLE01 = "Reconstruction quality vs training iterations — D-NeRF monocular scenes"
    TITLE27 = ("Post-peak behaviour: 4DGS native-4D peaks within the first thousands of "
               "iterations and then degrades")
    NOTE39 = ("Rows are ordered by the per-scene batch size of 4DGS-fudan (1, 2, 8, 24); the other two methods always use batch 1. "
              "Columns 2 and 3 are on a log axis because the sample and time budgets span an order of magnitude between methods. ")
    SAMPLE_NOTE = ("Every method is stopped after the same number of training images (30 000), which is the whole "
                   "Protocol-A budget of the batch-1 methods. Under this budget the partial 4DGS-fudan runs are complete: "
                   "at 30 000 samples T-Rex has run 1 250 of its 6 000 recorded steps and Jumping Jacks 1 875 of 15 000. "
                   "Lego stays unavailable for 4DGS-fudan. Values are linearly interpolated on the 1 000-iteration evaluation grid.")
    TIME_NOTE = ("Each scene is compared at the same wall-clock budget: the time the fastest method needed to finish its "
                 "30 000 steps on that scene (756 s to 2 284 s, always set by 4DGaussians). This is the budget a fixed "
                 "Colab T4 session actually buys. ")
    BSNOTE = ("Measured at the first crossing of the target, not at run termination. "
              "4DGS native-4D uses the per-scene batch sizes of its own repository (1-24), so its iteration "
              "counts are not directly comparable: see the training-samples figure (38).")
    NOTE43 = "The gap between the two panels is exactly the batch size of 4DGS-fudan on that scene. "
    GRID_NOTE = " Evaluation grid: every 1000 iterations."
    NOTE33 = {"4DGS-fudan": "Lego is not evaluable for this method (known limitation of the original paper). "}

else:
    DATASET = "Neural 3D Video (multi-view)"

    # Dynamic 3D Gaussians was part of this study until 2026-10-02 and was then set aside
    # (docs/METHODOLOGY_MULTIVIEW.md, section 1.1).
    METHOD_DIR = {
        "4dgaussian_n3dv_output": "4DGaussians (Wu et al.)",
        "4dgs_fudan_n3dv_output": "4DGS native-4D (fudan-zvg)",
        "spacetime_gaussians_output": "Spacetime Gaussians lite (Li et al.)",
    }
    # Spacetime Gaussians is benchmarked in both released variants, as two rows. Its runs
    # share one folder and are told apart by `stg_model` in the benchmark config.
    VARIANTS = {
        "spacetime_gaussians_output": ("stg_model", {
            "ours_lite": "Spacetime Gaussians lite (Li et al.)",
            "ours_full": "Spacetime Gaussians full (Li et al.)",
        }, "ours_lite"),
    }
    VARIANT_GROUPS = [["SpacetimeGS-lite", "SpacetimeGS-full"]]
    SHORT = {
        "4DGaussians (Wu et al.)": "4DGaussians",
        "4DGS native-4D (fudan-zvg)": "4DGS-fudan",
        "Spacetime Gaussians lite (Li et al.)": "SpacetimeGS-lite",
        "Spacetime Gaussians full (Li et al.)": "SpacetimeGS-full",
    }
    C = {"4DGaussians": BLUE, "4DGS-fudan": ORANGE,
         "SpacetimeGS-lite": AQUA, "SpacetimeGS-full": PURPLE}
    ORDER = ["4DGaussians", "4DGS-fudan", "SpacetimeGS-lite", "SpacetimeGS-full"]
    LABEL = {
        "4DGaussians": "4DGaussians (Wu et al.)",
        "4DGS-fudan": "4DGS native-4D (fudan-zvg)",
        "SpacetimeGS-lite": "Spacetime Gaussians lite (Li et al.)",
        "SpacetimeGS-full": "Spacetime Gaussians full (Li et al.)",
    }
    SCENES = ["coffee_martini", "cook_spinach", "cut_roasted_beef",
              "flame_salmon_1", "flame_steak", "sear_steak"]
    SCENE_LABEL = {"coffee_martini": "Coffee Martini", "cook_spinach": "Cook Spinach",
                   "cut_roasted_beef": "Cut Roasted Beef", "flame_salmon_1": "Flame Salmon",
                   "flame_steak": "Flame Steak", "sear_steak": "Sear Steak"}
    # No multi-view run exists yet; once the Protocol A loops have run, list here the
    # scenes completed by every method, exactly as the monocular study does.
    FAIR = list(SCENES)
    PARTIAL = set()
    NOT_EVAL = set()
    FAIRNOTE = "Aggregates are computed on the scenes completed by every method. "
    RADARNOTE = "Each axis is min–max normalised across the methods; it shows ranking, not absolute values."
    NMETHODS_WORD = "benchmarked"
    MEANNOTE = ("Mean over the scenes completed by every method; "
                "hollow markers are the individual scenes. ")
    NOTE36 = ""
    MISSING_SCENE_FILL = None
    HIGHLIGHT_SCENES = ["sear_steak", "flame_steak", "cook_spinach", "coffee_martini"]
    # 30 000 images is reached by every Protocol A run: the smallest is 4DGaussians on the
    # four scenes it trains with batch 2, 17 000 steps x 2 = 34 000 images.
    IMG_BUDGET = 30000

    GPU = "one GPU type for the whole study"
    # Protocol A is "each method at the budget its authors use" in this study, so there is
    # no step count common to the methods (STEP_BUDGET None drops the equal-steps views).
    PA_BUDGET = "official budget of each method"
    PA_BUDGET_CAP = "Official budget of each method"
    PA_RUNS = "official-budget runs"
    PA_AT = "at each method's official budget"
    PA_TOTAL = "(3 000 + 14 000 steps for 4DGaussians, 30 000 for the others)"
    STEP_BUDGET = None
    TITLE01 = "Reconstruction quality vs training iterations — Neural 3D Video multi-view scenes"
    TITLE27 = "Post-peak behaviour: PSNR relative to the peak of each run"
    NOTE39 = ("Columns 2 and 3 are on a log axis because the sample and time budgets differ "
              "widely between methods. ")
    SAMPLE_NOTE = ("Every method is read after the same number of training images (30 000), a budget every "
                   "Protocol A run reaches: the smallest is 4DGaussians at 17 000 steps x batch 2 = 34 000 images. "
                   "Values are linearly interpolated on the evaluation grid.")
    TIME_NOTE = ("Each scene is compared at the same wall-clock budget: the time the fastest method needed "
                 "to finish its own official budget on that scene. ")
    BSNOTE = ("Measured at the first crossing of the target, not at run termination. The methods use "
              "different batch sizes (2 or 4 views per step), so their iteration counts are not directly "
              "comparable: see the training-samples figure (38).")
    NOTE43 = "The gap between the two panels is the batch size of each method on that scene. "
    GRID_NOTE = " Evaluation grid: about 30 samples per run (every 500 or 1 000 iterations)."
    NOTE33 = {}

    _COMMON = ("cam00 held out, 1352x1014, LPIPS-VGG. Protocol A runs each method at its official "
               "N3DV budget, so the step axis is not a common unit of work: compare on images seen "
               "or on time. Spacetime Gaussians appears in the variants that were run (lite, full). ")
    if STUDY == "n3dv":
        # The main study: the whole sequence, as the three papers report it.
        NUM_FRAMES = 300
        CAVEAT = ("Multi-view study, main: Neural 3D Video, all 300 frames of each scene (300 test "
                  "views), " + _COMMON + "Spacetime Gaussians is trained as six 50-frame models per "
                  "scene, as in its paper: quality is the mean over the 300 test views, time, steps, "
                  "images and storage are summed over the six, VRAM is the maximum. No Protocol B "
                  "in this study.")
        # Summed images make a common image budget unequal for a method trained in blocks.
        SAMPLE_NOTE += (" Spacetime Gaussians' images are summed over its six models, so at a "
                        "common image budget each of its models has seen a sixth of it: read its "
                        "bar as a lower bound, and prefer the short-window study for this view.")
    else:
        # The short-window study: the first 50 frames, one Spacetime Gaussians block, and
        # the window Protocol B is run on.
        ANALYSIS_DIR = "analysis_f50"
        NUM_FRAMES = 50
        CAVEAT = ("Multi-view study, short window: Neural 3D Video, first 50 frames of each scene "
                  "(50 test views), " + _COMMON + "No published N3DV number is comparable with "
                  "these runs, which use a sixth of the sequence; see docs/METHODOLOGY_MULTIVIEW.md.")
