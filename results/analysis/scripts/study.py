"""Which study the pipeline is analysing: the monocular one or the multi-view one.

The two studies are reported separately (different datasets, different scenes,
different methods), so every table that names methods or scenes lives here and is
selected by the GS_STUDY environment variable:

    GS_STUDY=monocular   (default)  D-NeRF, three methods       -> results/
    GS_STUDY=n3dv                   Neural 3D Video, four       -> results/n3dv/

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
if STUDY not in ("monocular", "n3dv"):
    raise SystemExit("GS_STUDY must be 'monocular' or 'n3dv', not %r" % STUDY)

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
    RADARNOTE = "Each axis is min\u2013max normalised across the three methods; it shows ranking, not absolute values."
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

else:
    DATASET = "Neural 3D Video (multi-view)"

    METHOD_DIR = {
        "dynamic3dgaussians_output": "Dynamic 3D Gaussians (Luiten et al.)",
        "4dgaussian_n3dv_output": "4DGaussians (Wu et al.)",
        "4dgs_fudan_n3dv_output": "4DGS native-4D (fudan-zvg)",
        "spacetime_gaussians_output": "Spacetime Gaussians (Li et al.)",
    }
    SHORT = {
        "Dynamic 3D Gaussians (Luiten et al.)": "Dynamic3DGS",
        "4DGaussians (Wu et al.)": "4DGaussians",
        "4DGS native-4D (fudan-zvg)": "4DGS-fudan",
        "Spacetime Gaussians (Li et al.)": "SpacetimeGS",
    }
    C = {"4DGaussians": BLUE, "4DGS-fudan": ORANGE,
         "SpacetimeGS": AQUA, "Dynamic3DGS": PURPLE}
    ORDER = ["4DGaussians", "4DGS-fudan", "SpacetimeGS", "Dynamic3DGS"]
    LABEL = {
        "4DGaussians": "4DGaussians (Wu et al.)",
        "4DGS-fudan": "4DGS native-4D (fudan-zvg)",
        "SpacetimeGS": "Spacetime Gaussians (Li et al.)",
        "Dynamic3DGS": "Dynamic 3D Gaussians (Luiten et al.)",
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
    CAVEAT = ("Multi-view study: Neural 3D Video, first 50 frames of each scene, cam00 held "
              "out (50 test views), 1352x1014, LPIPS-VGG. The 50-frame window means no "
              "published N3DV number is comparable with these runs; see "
              "docs/METHODOLOGY_MULTIVIEW.md. Dynamic 3D Gaussians optimises frame by frame, "
              "so its iteration axis is not the same unit of work as the other three.")
    FAIRNOTE = ("Aggregates are computed on the scenes completed by all four methods. ")
    RADARNOTE = "Each axis is min\u2013max normalised across the four methods; it shows ranking, not absolute values."
    NMETHODS_WORD = "four"
    MEANNOTE = ("Mean over the scenes completed by all four methods; "
                "hollow markers are the individual scenes. ")
    NOTE36 = ""
    MISSING_SCENE_FILL = None
    HIGHLIGHT_SCENES = ["sear_steak", "flame_steak", "cook_spinach", "coffee_martini"]
    # 60 000 images is reached by all four at the end of Protocol A: the smallest
    # is Spacetime Gaussians at 30 000 steps x batch 2.
    IMG_BUDGET = 60000
