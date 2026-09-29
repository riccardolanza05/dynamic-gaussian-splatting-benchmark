# Open decisions, multi-view study

Six choices in the multi-view extension are **judgement calls, not facts**. Five are implemented with a default, or need no code at all, so they do not block a run. The GPU has no default, because it is chosen when the machine is started. Each changes what the resulting numbers mean, and the defaults were chosen by the person who wrote the notebooks, not by the person who owns the study.

This page exists so the decision can be taken later, with the trade-off in front of you, rather than discovered when the tables are already built. It starts with a short refresher on what the study measures and how the four methods differ, because every decision below follows from those differences. Each entry then says what is implemented today, what the alternatives are, what each one costs, and what the resulting number would mean. A [reference section](#what-the-papers-and-repositories-say-checked-2026-09-29) at the end collects the facts the entries rely on, each with its source.

Status of all six: **not yet decided.** Decisions 1, 3, 5 and 6 are a knob or an analysis step, and cost at most one notebook to rerun, so they can even be taken after the first results. **Decisions 2 and 4 are expensive to change afterwards — changing either means rerunning everything — so read those first.**

| # | Decision | Current default | Cost of changing later |
|---|---|---|---|
| [1](#1-iteration-budget-of-dynamic-3d-gaussians) | Protocol A budget of Dynamic 3D Gaussians | `"native"` | one knob, rerun that notebook |
| [2](#2-temporal-window-50-frames) | Temporal window, and whether to add a 300-frame verification run | 50 frames, no verification run | main window: **everything must be rerun**; verification run: can be added at any time |
| [3](#3-which-spacetime-gaussians-model) | Spacetime Gaussians variant | `ours_lite` | one knob, rerun that notebook |
| [4](#4-which-gpu) | Which GPU the whole study runs on | none — **must be chosen before the first run** | **everything must be rerun** |
| [5](#5-how-to-configure-dynamic-3d-gaussians-on-n3dv) | How Dynamic 3D Gaussians is configured on N3DV (masks, floor loss) | estimated masks + floor loss | code change + rerun notebook 04 |
| [6](#6-the-protocol-a-budget-of-30-000-steps) | The Protocol A budget, and whether to also read results at each method's official budget | 30 000 uniform, no official readout | none for the readout (analysis only) |

---

## Refresher: what is measured, and how the four methods differ

### The two protocols

Every method is scored on two axes: **quality** (PSNR, SSIM, LPIPS and L1 on the held-out views) and **cost** (training time, peak VRAM, model size on disk, rendering speed). The two protocols of the monocular study are kept unchanged:

* **Protocol A — fixed budget.** Every method trains for the same number of optimisation steps and the quality it reaches is compared. It answers *"for the same effort, which method reconstructs best?"*.
* **Protocol B — fixed quality.** Every method trains until its test L1 drops below a per-scene target, and what it cost to get there is compared. It answers *"what does this quality cost?"*. The targets are derived from the Protocol A results, so Protocol B always runs after Protocol A.

### The dataset

**N3DV** (*Neural 3D Video*): about 20 fixed, synchronised cameras film a kitchen with a person moving in it (cooking, a flame, a salmon…). Each scene is 300 frames, 10 seconds at 30 fps. Camera `cam00` is held out for testing.

### The four methods, in one line each

| Method | How it represents motion | Consequence for this page |
|---|---|---|
| **4DGaussians (Wu)**, notebook 05 | One set of "canonical" 3D Gaussians plus a small network (HexPlane grid + MLP) that, given a time *t*, says how much to move, rotate and scale each Gaussian. | **One model for the whole sequence.** |
| **4DGS native-4D (Fudan)**, notebook 06 | The Gaussians are four-dimensional: they have an extent in time too. Rendering instant *t* means "slicing" each 4D Gaussian into a 3D one. | **One model for the whole sequence.** |
| **Spacetime Gaussians (Li)**, notebook 07 | Each Gaussian has an opacity that rises and fades over time (a bell curve in time) and a polynomial trajectory. | One model, but **by construction it covers a block of 50 frames** — the root of decision 2. |
| **Dynamic 3D Gaussians (Luiten)**, notebook 04 | Reconstructs the first frame well, then **walks the sequence one frame at a time**: each new frame starts from the Gaussians of the previous one and optimises only their position and rotation, under "physical" constraints (local rigidity, isometry) that make them behave like solid objects. Colour and size stay fixed. It is **tracking**, not one model optimised all at once. | Its cost is spent **per frame** — the root of decision 1. |

---

## 1. Iteration budget of Dynamic 3D Gaussians

**Where:** cell 0.1 of [`notebooks/04_dynamic3dgaussians_luiten_n3dv.ipynb`](../notebooks/04_dynamic3dgaussians_luiten_n3dv.ipynb), the variable `D3DG_BUDGET_MODE`.

### The problem

For notebooks 05, 06 and 07, "30 000 steps" means the same thing: 30 000 updates of a single model that sees the whole window. Dynamic 3D Gaussians spends its steps **per frame** instead. Its authors use 10 000 steps on the first frame (building the scene from scratch) and 2 000 on each later one (following the motion). Its total is therefore a *function of the window*, not a number anyone chose:

```
total = ITERS_FIRST + ITERS_PER_TIMESTEP × (NUM_FRAMES − 1)
      = 10 000 + 2 000 × 49 = 108 000   (at 50 frames)
```

That is 3.6× the budget of the other three. It is not a choice of ours: it is the shape of the algorithm. No setting of that method both equals 30 000 total steps and leaves its per-frame budget alone.

### Option A — `"native"`, 108 000 steps (implemented)

`D3DG_ITERS_FIRST = 10000`, `D3DG_ITERS_PER_TIMESTEP = 2000`: the authors' schedule, unchanged.

**Pros**

* It measures the method **as its authors designed it**. Whether it does well or badly, that is to its credit or blame, not to a wrong budget.
* It follows the precedent of the monocular study. There, 4DGS native-4D used a different batch per scene (1 to 24 images per step) and it was not forced to a common value: the difference was declared and analysed on the `images_seen` axis (see [METHODOLOGY.md](METHODOLOGY.md) §3.5 and [RESULTS.md](RESULTS.md) §4). The same argument applies here, more strongly: the per-frame budget is not a tuning knob but the shape of the algorithm.

**Cons**

* In the Protocol A table the "steps" column is no longer equal for everyone: 108 000 against 30 000. A hurried reader may conclude that this method "was given more resources". In part that is true, and it has to be written next to the number.
* It is **by far the longest run** of the four notebooks: 3.6× the steps of the others, for six scenes, and the one whose cost is most sensitive to decision 4. The loop is resumable per scene, but not mid-scene.

### Option B — `"aligned"`, 30 000 steps in total

`D3DG_BUDGET_MODE = "aligned"` keeps the first frame at 10 000 and divides the remaining `ALIGNED_TOTAL_ITERATIONS − 10 000` equally over the later frames: at 30 000 total and 50 frames, **about 408 steps per frame instead of 2 000** — a fifth.

**Pros**

* The Protocol A table is literally clean: 30 000 steps for everyone, so "at equal steps, which reconstructs best?" has a literal answer.
* Much less GPU time.

**Cons**

* 408 steps per frame are probably too few for the tracking to settle. The Gaussians lag behind the real motion, and the error **accumulates**, because every frame starts from the previous one; the persistence losses (rigidity, rotation, isometry, background anchoring) have that much less opportunity to act before the window moves on.
* The resulting number says more "a starved Dynamic 3D Gaussians" than "Dynamic 3D Gaussians". It is very likely a floor, not a measurement of the method, and risks concluding that the method is poor when it was simply run outside its design point.

### Option C — both

The knob makes this cheap in code and expensive in GPU time: one extra full loop of notebook 04. The run folders already carry the budget in their name (`<scene>_f50_iters108000` against `<scene>_f50_iters30000`), so the two coexist without collision, and `budget_mode` is recorded in every benchmark JSON. In the tables, *native* is the headline result and *aligned* a footnote ("at strictly equal steps, the method drops to X dB").

**Pros:** the most defensible outcome — it answers both questions.
**Cons:** one more complete run of notebook 04, which is already the longest.

### A note on Protocol B, whatever the option

Protocol B stops training as soon as the target quality is reached. For this method stopping early does not mean "a less refined model", it means **"a truncated sequence"**: say 30 frames out of 50 reconstructed, and the rest does not exist. For this method Protocol B answers a different question, and Protocol A remains the primary comparison (see [DISCLOSURES_MULTIVIEW.md](DISCLOSURES_MULTIVIEW.md) §5). None of the three options changes that.

**Recommendation:** A as the baseline, C if the GPU time allows it. B alone, no.

---

## 2. Temporal window: 50 frames

**Where:** cell 0.1 of all four notebooks, the variable `NUM_FRAMES`.

### What the window is

Each N3DV scene is a **10-second video at 30 fps, i.e. 300 frames**, filmed by about 20 synchronised cameras, one of which (`cam00`) is never used for training and serves only as the test camera. The *window* is how many consecutive frames each method is asked to reconstruct: all 300 (the full 10 seconds), or the first 50 (1.67 seconds). The test is always `cam00` inside the window, so 50 or 300 test images.

The window changes three things: **what the experiment costs**, **how hard the problem is**, and above all **what the results can be compared with**.

### How each method reacts to a longer window

This is the crux, because the four methods react in very different ways.

* **4DGaussians (Wu) and 4DGS native-4D (Fudan)** build **one model for the whole window**, with a fixed number of steps (3 000 + 14 000 and 30 000 officially). A longer window **does not add steps**, but it changes two things:
  * **each image is seen fewer times.** Wu's 14 000 main steps at 2–4 images per step see about 28 000–56 000 images. At 50 frames × ~20 cameras there are about 1 000 training images, each revisited on average **30–55 times**; at 300 frames about 6 000, each seen only **5–10 times**;
  * **the same model capacity has to represent more motion.** Wu's deformation grid has a fixed size; Fudan's 4D Gaussians have to cover a longer time span, so it tends to need more of them, and each step becomes somewhat slower.

  For these two, 300 frames cost **about the same number of steps**, but are a **harder problem**.
* **Spacetime Gaussians (Li)** is designed around **blocks of 50 frames** (`duration: 50` in every official config). To cover 300 frames its authors train **six independent models**, one per block, and report the **average** of the six. At 300 frames it costs six times as much, and the result is a set of six models, not one.
* **Dynamic 3D Gaussians (Luiten)** walks the sequence **one frame at a time**, at a fixed cost per frame (2 000 steps after the first frame's 10 000). Its cost is proportional to the window: **108 000 steps at 50 frames, 608 000 at 300** — impractical on a rented GPU, times six scenes.

And there is **disk**: 300 frames means extracting and holding six times as many images. One 1352×1014 PNG is roughly 2–3 MB, so 50 frames × ~20 cameras is on the order of **2–3 GB** per scene and 300 frames **13–18 GB**, before any per-method copies (Spacetime Gaussians, for instance, lays out one COLMAP folder per frame). An earlier version of this page said 15–20 GB for 50 frames; that figure looks overstated and is to be measured in the smoke test.

**In short:** 50 frames is the longest window in which **all four** methods run on the same machine in reasonable time, and it is exactly one Spacetime Gaussians block, so no method is split into pieces.

### Why 50 frames breaks comparability with the papers

Every paper reports its N3DV results **on 300 frames**. Our PSNR at 50 frames and theirs at 300 measure different things, for three reasons:

1. **The test set is different**: 50 images of `cam00` for us, 300 for them. If the first 1.67 seconds of a scene are easier or harder than the rest (the flame of `flame_salmon` not yet lit, less motion), the average changes even for an identical method.
2. **The problem is easier**: as above, at equal steps every image is revisited far more often and there is less motion to represent. PSNR **higher** than the papers' is expected — by how much is unknown.
3. **For Spacetime Gaussians the comparison is nearly, but not quite, like for like**: our run is the first of the paper's six blocks; the paper reports the average of the six.

**What is lost in practice.** In the monocular study the comparison with the papers was the **quality check** ([METHODOLOGY.md](METHODOLOGY.md) §7): 40.64 dB against 41.01 published for Deformable-3DGS, and an anomalous gap on 4DGaussians that revealed the white/black background issue. Without it, an error **common to all methods** — camera poses converted wrongly, a wrong test split, a different resolution — would go unnoticed: every method would lose, say, 1–2 dB, the ranking could still look plausible, and nobody would see it. At 50 frames the results remain **comparable with each other**, since all four are under the same conditions, but not **verifiable from outside**.

### Option A — 50 frames for everyone (implemented)

* **Pros:** one window, one protocol, minimum cost.
* **Cons:** no external check. Every table has to state that the numbers are not comparable with the literature (already written into [METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md) §2.3 and §8).

### Option B — 50 frames for everyone, plus a 300-frame verification run of Wu and Fudan only

The main comparison table stays at 50 frames. The 300-frame run serves **only as a check**: our Wu and Fudan are compared with the numbers in their papers.

* **Condition for it to be a real check:** the verification run has to be read **at each method's official budget** (Wu at 14 000 fine steps, Fudan at 30 000); otherwise a gap cannot say whether the pipeline or the budget is different. That reading is already on the Protocol A curve and costs no GPU time (decision 6, option C).
* **What it actually verifies:**
  * the **common** part: frame extraction, poses, `cam00` as the test camera, resolution, metric code;
  * the **Wu- and Fudan-specific** part.

  It **does not verify** the preparation specific to the other two: Spacetime Gaussians runs its own per-frame COLMAP reconstruction, and Dynamic 3D Gaussians has its own conversion to the Panoptic format. For those two, confidence rises only partly. (An earlier version of this page was more optimistic on this point.)
* **Wu allows a scene-by-scene check**, because its paper has a per-scene table (supplementary Table 6), as in the monocular study. Fudan reports only the six-scene average.
* **Cost: less than it sounds.** Wu and Fudan take **the same number of steps** at 50 or 300 frames. The extra cost is mostly **disk and preparation** (extracting six times as many frames), plus a per-step slowdown of Fudan that has to be measured. `DELETE_FRAMES_AFTER_TRAIN` frees each scene after its run, so it can be done one scene at a time.
* **Cons:** the 300-frame numbers must not be mixed with the 50-frame ones. They are a control experiment, not an extra column of the table.

### Option B+ — as B, plus Spacetime Gaussians at 300 frames "as in the paper"

Six 50-frame blocks, averaged as the paper does. It would work for STG too, but it costs **six times the whole STG loop** and needs new code: today the notebook trains only the first block. Listed for completeness; worth it only if B reveals problems.

### Option D — a second, full-length study: 300 frames for the three methods that can do it

Proposed by the study owner (2026-09-29). Two comparisons instead of one:

* **short window, all four methods**: the main table of options A/B, 50 frames;
* **full sequence, three methods**: 4DGaussians, 4DGS native-4D and Spacetime Gaussians on all 300 frames, **as their papers do**; Dynamic 3D Gaussians is left out, because at 300 frames it is impractical (608 000 steps per scene). Deformable-3DGS, the fifth studied method, is monocular only and is in neither table, so this is **three of the four multi-view methods**.

Spacetime Gaussians runs at 300 frames exactly as its paper does: **six independent models of 50 frames**, one per block. The quantities then need a stated definition, the paper's where it has one:

* **quality** (PSNR, SSIM, LPIPS, L1): the average over the 300 test images, i.e. over the six blocks, as the paper reports it;
* **training time**: the sum of the six trainings;
* **storage**: the sum of the six models;
* **peak VRAM**: the maximum of the six.

**Pros**

* It is the literature check of option B, extended to Spacetime Gaussians: each of the three is directly comparable with its own paper, scene by scene for 4DGaussians.
* It is a result in its own right: the three methods on the task as the field defines it, not only on a short window.
* For 4DGaussians and 4DGS native-4D the cost in steps is the same as at 50 frames; the extra cost is disk and preparation.

**Cons**

* **Spacetime Gaussians costs six times its 50-frame loop**, and its preparation grows from 50 to 300 per-frame COLMAP reconstructions per scene (CPU time).
* **It needs new code in notebook 07**: training block *k* on frames 50k … 50k+49, and merging the six results into one benchmark entry per scene. Today the notebook trains only the first block. It also needs a small change in the analysis, which must not mix the 300-frame table with the 50-frame one.
* Protocol B at 300 frames is not proposed: its targets would have to be recalibrated, and for Spacetime Gaussians "the first time the target is reached" is not defined across six separate models. The second study would be Protocol A only, read at each method's official budget (decision 6, option C).

**Status:** the owner's current preference, not yet implemented. It subsumes option B.

### Option C — 300 frames for everyone

Not feasible: Dynamic 3D Gaussians becomes impractical and Spacetime Gaussians becomes six models. It would mean dropping one or two methods from the comparison.

### When the decision has to be taken

An earlier version of this page said this decision "has to be taken first and costs rerunning everything". More precisely:

* **Only the window of the main table is irreversible.** If all runs are made at 50 frames and later the main comparison is wanted at 300, everything is rerun.
* **The verification run of option B can be added at any time.** It invalidates nothing: it goes into separate folders (`_f300_`), and the notebooks refuse to mix windows (`run_is_complete()` checks `num_frames`).

In practice: **50 or 300 for the main table must be chosen first**, and 50 is effectively forced if all four methods are to stay. **A or B can wait for the smoke test**, when the disk and time costs are known.

**Recommendation:** 50 frames for the main table, and add B as soon as the smoke test confirms that disk and time allow it.

---

## 3. Which Spacetime Gaussians model

**Where:** cell 0.1 of [`notebooks/07_spacetime_gaussians_li_n3dv.ipynb`](../notebooks/07_spacetime_gaussians_li_n3dv.ipynb), the variable `STG_MODEL`.

### The two released models

The authors release two, with separate configs (`configs/n3d_lite/` and `configs/n3d_full/`) and separate rasterizers. They differ in **how the Gaussians carry colour**:

* **`ours_lite`** — each Gaussian directly holds an RGB colour, and the rasterizer (`diff_gaussian_rasterization_ch3`) writes it into the image.
* **`ours_full`** — each Gaussian holds 9 numbers, a *feature* rather than a colour. The rasterizer (`diff_gaussian_rasterization_ch9`) produces a feature image, and a **small neural network** (an MLP, `rgbdecoder`) turns it into RGB, taking view direction and time into account. It renders reflections and view-dependent effects better, and it is the variant the paper's headline table reports.

### Option A — `ours_lite` (implemented)

**Pros**

* **Homogeneity.** None of the other three methods has a neural colour decoder: 4DGaussians and 4DGS native-4D use spherical harmonics, Dynamic 3D Gaussians an RGB per Gaussian. With lite all four produce colour "inside the Gaussians", and the VRAM, time and storage columns measure the same thing.
* Cheaper per step, and fits a smaller GPU more easily.

**Cons**

* The paper's headline row is `Ours` (full), not `Ours-lite`. A reader who knows the paper expects the higher number. In the paper's own N3DV table the gap is about **0.5 dB of PSNR**: small but real.

### Option B — `ours_full`

**Pros**

* The method is represented at its best, as its authors present it, and its number sits next to the paper's headline row (32.05 dB against 31.59 for lite, on the 300-frame average).
* **It works today with no change to the authors' code.** An earlier version of this page claimed that the released `save_ply()` does not write the decoder. That was wrong: the commented-out `torch.save` is in `ourslite.py`, where there is no decoder to save. In `oursfull.py` the decoder has been saved as `point_cloud.pt` since the first commit (`oursfull.py:411`) and every loader reads it back (`:483`, `:572`, `:661`, `:770`). Notebook 07 already counts `point_cloud.pt` in the storage column, and the training renderer applies the decoder itself (`renderer/__init__.py:103`), so the monitor's metrics are those of the full model. See [DISCLOSURES_MULTIVIEW.md](DISCLOSURES_MULTIVIEW.md) §8.

**Cons**

* More VRAM and more time per step than lite (a 9-channel rasterizer plus the MLP).
* One method out of four has a neural component the others lack. If it wins on PSNR, it is not clear whether the credit goes to the spacetime representation or to the network. The authors themselves say the MLP "mainly compensates for view-dependent appearance changes" and that even subtle colour mismatches move the PSNR (issue [#119](https://github.com/oppo-us-research/SpacetimeGaussians/issues/119)), so part of the gap is exactly the kind of per-view colour correction that Dynamic 3D Gaussians does with its per-camera gain and offset, and the other two do not do at all.

### Option C — both

`STG_MODEL` is a knob (`--set STG_MODEL=ours_full`), `ours_full` runs go to their own folders (`<scene>_f50_iters30000_full`), and `stg_model` is in every JSON. One extra loop of notebook 07, with its per-frame COLMAP stage cached and shared. **Not yet done:** the analysis pipeline treats one method folder as one method, so with both variants in it the tables would mix them. For C, the analysis has to be taught to split them into two rows — a small change, to be made only if C is chosen. The table would carry `full` as the method's headline, as in the paper, and `lite` as the homogeneous comparison.

### What to weigh

Do you want **the methods as their authors present them** (B), or **the methods reduced to a common shape so the columns mean the same thing** (A)? The monocular study consistently chose the second: one resolution, one background, one budget, one LPIPS backbone. There is no longer a technical obstacle to either.

**Recommendation:** B, given the stated goal of staying comparable with the papers, with the decoder declared in the methodology as a difference that is kept, not removed. C if GPU time allows, because `lite` is the variant that answers the homogeneity question. The earlier recommendation of A rested on the non-existent bug, and is withdrawn.

---

## 4. Which GPU

**Where:** not in the code at all — it is what you select when you start the machine. See also the [running notes in the README](../README.md#running-on-a-cloud-gpu-over-ssh).

### Why this is a decision and not a detail

Two of the ten monitored metrics, **training time** and **peak VRAM**, are not properties of the method but of the pair *method + card*. The same run is much faster on an L4 than on a T4. Therefore:

* **every run must use the same GPU type**, including any rerun of a single scene. Mixing cards makes the times incomparable and removes the meaning of Protocol B;
* the benchmark JSON **does not record** which card produced a number, so it has to be written down in the documents;
* **one run at a time** on the GPU, because VRAM is measured on the whole device (by polling `nvidia-smi`) and a second run would be charged to the method under test.

> If two methods are trained on two different GPUs, their training times cannot be compared, and the "quality per unit of cost" question that Protocol B exists to answer has no answer. **Mixing GPU types means rerunning.**

### The options

| GPU | Pros | Cons |
|---|---|---|
| **T4** (16 GB) | The same card as the monocular study: in principle both studies' costs sit on one scale. Cheapest per hour. | The slowest, and the multi-view methods are heavier than the monocular three; Dynamic 3D Gaussians at 108 000 steps × 6 scenes becomes very long. The "continuity" is also **partly illusory**: dataset, resolution and window differ, and so does the machine around the GPU (CPU and disk, which matter for COLMAP and data loading) compared with Colab. |
| **L4** (24 GB) | Good speed/price balance and VRAM headroom. Makes the "expensive" options (1C, 2B) affordable. | Times are not comparable with the monocular study. |
| **A10G** (24 GB) | Faster still. | More expensive per hour; same incomparability. |
| **A100** (40/80 GB) | The fastest. | Wasteful: in the monocular study no run exceeded 5.1 GB of VRAM. You pay for memory you do not use. |

The binding constraint is mostly **time**, not memory — with one exception. Spacetime Gaussians' README states that training on N3DV needs 24 GB, because the ground-truth images are held on the GPU as floats. At 50 frames that is roughly 15–16 GB of images, which does not fit a T4. The repository's `--gtisint8 1` option stores them as 8-bit integers, which is lossless for 8-bit PNG frames. On a T4 it is required (`--set EXTRA_TRAIN_ARGS="--gtisint8 1"` for notebook 07); on a 24 GB card it is optional.

### What can honestly be said about cost today

No estimate of GPU hours is given here, on purpose: nothing has run yet, and extrapolating from papers that used other GPUs, other windows and other budgets would give a number that looks precise and is not. What the papers do report, as anchors only:

| Method | Published training time on N3DV | Conditions |
|---|---|---|
| 4DGaussians | 40 min per scene | 300 frames, 3 000 + 14 000 steps (paper Table 3; GPU not stated in the N3DV setup) |
| Spacetime Gaussians | 40–60 min per 50-frame chunk | NVIDIA A6000 (paper, Appendix) |
| 4DGS native-4D | not reported | — |
| Dynamic 3D Gaussians | not reported on N3DV | — |

**The way to get real numbers is a smoke test**, on the GPU type being considered. It is cheap, and it is needed anyway to check that the pipeline runs:

1. one scene (`sear_steak`), `NUM_FRAMES=50`, Protocol A, but stopped after about 1 000–2 000 steps per method, through `run_benchmark.py`;
2. read from each JSON the seconds per step and the peak VRAM, and from the log the preparation time (download, frame extraction, COLMAP; notebook 07's 50 per-frame COLMAP runs are CPU time and do not depend on the GPU);
3. multiply: seconds per step × budget × 6 scenes, plus preparation, for each option of decisions 1, 2 and 3.

The result is a table of hours per option on that card, which is what the budget discussion with the project's collaborators needs. Run the same smoke test on two card types (for example T4 and L4) if the choice between them is open: a few dollars of GPU time buy a decision that cannot be reversed later.

**Recommendation:** an L4, stated wherever the results are quoted — to be confirmed by the smoke test and by the budget available.

---

## 5. How to configure Dynamic 3D Gaussians on N3DV

**Where:** notebook [`04`](../notebooks/04_dynamic3dgaussians_luiten_n3dv.ipynb), cells 3.5 (masks) and the world-frame conversion; see [METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md) §5.2.

### The problem

The authors of Dynamic 3D Gaussians **never ran it on N3DV**: their paper uses their own preparation of CMU Panoptic Sports, and their data-preparation code is not released (README, "Partial code release"). The repository's issues contain no guidance for N3DV from the authors ([#18](https://github.com/JonathonLuiten/Dynamic3DGaussians/issues/18) and [#17](https://github.com/JonathonLuiten/Dynamic3DGaussians/issues/17) cover custom data in general: "a point cloud from colmap should be fine").

The method needs two things N3DV does not have: a **foreground/background mask per image** (its `seg` loss, and the split that decides which Gaussians get the rigidity losses and which are anchored as background), and a **known floor plane** (its floor loss). Notebook 04 currently builds both: masks estimated from each camera's temporal median, and a world frame rotated and translated so that the floor sits at y = 0 ([DISCLOSURES_MULTIVIEW.md](DISCLOSURES_MULTIVIEW.md) §2).

### What exists: the only published N3DV number, and how it was made

The only published result of Dynamic 3D Gaussians on N3DV is in the **Spacetime Gaussians paper** (Appendix B, Table 6): **30.67 dB PSNR, 0.099 LPIPS**, on the six scenes, 300 frames. Later papers that quote it (for example HiCoM, NeurIPS 2024) take it from there. The STG authors write that the default hyper-parameters gave "subpar" quality on N3DV and that they tuned them. In issue [#81](https://github.com/oppo-us-research/SpacetimeGaussians/issues/81) of their repository, a maintainer of the repository (`lizhan17`, marked as collaborator on GitHub) states what they changed:

> "Since the neural 3D video dataset does not provide foreground/background masks, we turn off the background segmentation loss in Dynamic3DGaussians and just set the last column of init_pt.cld.npz as all ones. We also disable the floor loss and set the log_scales in initialize_optimizer to 0.005."

In plain terms: **no masks at all** (every Gaussian is treated as foreground, so the rigidity, rotation and isometry losses act on all of them and nothing is anchored as background), **no floor loss**, and the learning rate of the Gaussians' scales raised from 0.001 to 0.005 (`train.py:71` sets it in the optimiser; it is a learning rate, not an initial scale).

### Option A — estimated masks + floor loss (implemented)

**Pros**

* Keeps every component of the method as designed, including the static/dynamic split, which is what lets it pin the kitchen in place and spend its tracking on the person.

**Cons**

* The masks are **this project's invention**. Their quality bounds the method's result, and the method's own README calls even the authors' masks "REALLY bad" and says they visibly degrade the results. Our masks may be better or worse, and nobody else has used them.
* No published number was made this way, so there is nothing to check the result against.
* The `seg` loss costs a second rendering pass per step (README, "Speeding up the code").

### Option B — the STG authors' recipe

**Pros**

* It is the **only configuration with a published N3DV result**, so it is the one closest to comparability with the literature — the stated goal of the study.
* It removes the part of notebook 04 that is most clearly ours (the masks) and the floor alignment, which was inferred rather than known. Less of the result depends on this project's preparation.
* One rendering pass per step instead of two.

**Cons**

* It is not the method as its authors designed it, but as another group tuned it for this dataset. That has to be declared.
* **The comparability is limited even so.** The published 30.67 dB is on 300 frames, and whether STG's authors ran it as one sequence or in 50-frame chunks is not stated. At 50 frames our number is not directly comparable (decision 2).
* The reporter of issue #81 got NaN losses applying the recipe, and the cause appears to have been the camera-pose conversion, not the recipe. Notebook 04's conversion is untested on a GPU, so a **1-frame smoke test** (`NUM_FRAMES=1`: static reconstruction of the first frame only) should come first whichever option is chosen.
* It needs a code change in notebook 04, not only a flag: a knob (`D3DG_SEG_MODE`) that writes an all-ones `seg` column, switches off the `seg` and `floor` losses, and sets the scale learning rate. Two details make it more than a switch:
  * with every Gaussian marked as foreground the background set is empty, and the `bg` loss (`helpers.l1_loss_v2`, a `.mean()` over the background points) would be the mean of an empty tensor, i.e. NaN. The `bg` term has to be dropped too, which is what the quote's "background segmentation loss" most likely means;
  * `get_dataset` (`train.py:23`) still opens a `seg/*.png` for every image, so constant masks must still be written.

  A few hours including a 1-frame test; not done yet.

### Option C — both

The knob would make this possible, at the cost of one more loop of notebook 04, the most expensive notebook.

**Recommendation:** B, as the default, given the goal of staying as close as possible to published practice; A as a secondary run only if GPU time allows. The recommendation can change if the 1-frame smoke test shows the recipe failing on our conversion.

---

## 6. The Protocol A budget of 30 000 steps

**Where:** `MAX_ITERATIONS` in cell 0.1 of notebooks 05–07 (notebook 04 has its own budget, decision 1).

### Where 30 000 came from, and whether it still holds

In the monocular study 30 000 was a representative value of the three official D-NeRF schedules: 40 000 for Deformable-3DGS, 3 000 + 20 000 for 4DGaussians, 30 000 for 4DGS native-4D. On N3DV the official schedules are different:

| Method | Official N3DV schedule | Batch (views per step) | Protocol A today | Ratio |
|---|---|---|---|---|
| 4DGaussians | 3 000 coarse + **14 000** fine | 4, or 2 on four scenes | 3 000 + 27 000 | **≈ 1.9× the official fine budget** |
| 4DGS native-4D | **30 000** | 4 | 30 000 | 1× |
| Spacetime Gaussians | trains **30 000**, the N3DV configs evaluate the **25 000** snapshot | 2 | 30 000 | 1× (1.2× the evaluated snapshot) |
| Dynamic 3D Gaussians | 10 000 + 2 000 per frame | 1 | see decision 1 | — |

So the intuition that 30 000 no longer fits is **half right**. It is still the median and exactly the official budget of two methods, but it runs 4DGaussians at nearly twice its official N3DV budget. On D-NeRF it was 23 000 official against 30 000; on N3DV it is 17 000 against 30 000.

### A finding that makes this cheaper than it looks

In all three whole-window methods the **learning-rate schedules are fixed in steps, not in proportion to the total budget**:

* 4DGaussians: every scheduler uses `position_lr_max_steps = 20 000` (`arguments/__init__.py:119`, `scene/gaussian_model.py:186–196`). Densification stops at `densify_until_iter = 10 000` (`arguments/dynerf/default.py`). `opt.iterations` is only used to stop, to save, and to skip the last optimiser step (`train.py:130, 238, 290`).
* 4DGS native-4D: `position_lr_max_steps: 30_000` in each official YAML.
* Spacetime Gaussians: `position_lr_max_steps = 30 000`, densification until 9 000 (`arguments/__init__.py:98, 118`).

Consequence: a run with a longer budget passes **through the same states** as a run with the official budget, up to randomness. The 4DGaussians model at fine step 14 000 of a 27 000-step run is, in practice, the official N3DV model; the Spacetime Gaussians model at step 25 000 is the one its authors evaluate. Protocol A already samples the metrics every 1 000 steps, so **the official-budget result is already on the curve, at no extra GPU cost**.

### Option A — 30 000 uniform, as the monocular study (implemented)

**Pros:** one budget for three methods, the same rule as the monocular study; "at equal steps, which is best?" has a literal answer.
**Cons:** 4DGaussians is run well past its design point. On D-NeRF it kept improving past its budget, but that is not guaranteed on N3DV. The headline number is then not the one its authors would report.

### Option B — each method at its official budget

**Pros:** each method's number is the one its authors would produce, which is what comparability with the papers needs.
**Cons:** the step column differs (17 000, 30 000, 30 000). And "steps" were never an equal unit of work anyway, because the batches differ: `images_seen` is the axis the analysis already uses to compare budgets.

### Option C — keep 30 000, and also read each method at its official budget

The same runs as A, plus one analysis step: a second table that takes, for each method, the entry at its official budget (4DGaussians fine 14 000, Spacetime Gaussians 25 000, 4DGS native-4D 30 000). No GPU cost.

**Pros:** both questions answered from one set of runs — "at equal steps" (A) and "as the authors run it" (B). This is also the number to set next to the papers in the 300-frame validation of decision 2B.
**Cons:** two tables to explain instead of one. There is one technical caveat to check on the first real run: the monitor's renderer is the training renderer, not always the exact pipeline of each repository's `test.py` (Spacetime Gaussians' test uses a fused forward-only rasterizer). Small differences of the order of rounding are possible; they are the same in both tables, so they do not affect the comparison between them.

**Recommendation:** C. It costs nothing, and it gives the "official budget" number without giving up the homogeneous one.

---

## How the decisions interact

* **1C, 2B, 3C and 5C all add GPU hours.** How many of them are affordable is exactly what the smoke test of decision 4 has to establish.
* **The irreversible ones are the main window (2) and the GPU (4).** The 300-frame verification run of 2B, and decisions 1, 3, 5 and 6, are a knob, one notebook to rerun, extra runs in separate folders, or an analysis step, and can be decided after the first results.
* **2B and 6C belong together.** The 300-frame validation run is only a real check if it is read at the official budget, and 6C is what provides that reading.

Suggested combination if the budget allows it: **1 = native (+ aligned), 2 = 50 frames + the 300-frame run of 05 and 06, 3 = full (+ lite), 4 = L4 after a smoke test, 5 = the STG recipe, 6 = C.** Minimal but still defensible: **native, 50 frames, full, L4, STG recipe, 6C**, with the lack of a literature check at 50 frames declared.

## Preliminary orientation of the study owner (2026-09-29, not final)

Recorded so the next steps follow it; each point may change with more information.

1. **C** if the budget allows, otherwise **A**. Needs a GPU-hour estimate, from a smoke test.
2. **D** (a second study at 300 frames for 4DGaussians, 4DGS native-4D and Spacetime Gaussians, without Dynamic 3D Gaussians), otherwise **B**, otherwise **A**, depending on the budget.
3. Preference for **`ours_full`**, provided it works as the paper specifies. It does: the decoder-save "bug" did not exist (see decision 3 and [DISCLOSURES_MULTIVIEW.md](DISCLOSURES_MULTIVIEW.md) §8).
4. To be decided after smoke tests and a budget/access discussion with the project's collaborators.

A general goal stated alongside: **results should stay comparable with the papers as far as possible**, while accepting the trade-offs a homogeneous comparison needs.

---

## What the papers and repositories say (checked 2026-09-29)

### Which methods have official N3DV results

The project studied five methods. Four accept multi-camera input; Deformable-3DGS is monocular only and has no N3DV path.

| Method | Official N3DV result | Reported by the authors | Settings behind the number |
|---|---|---|---|
| 4DGaussians (Wu et al.) | **yes** — 31.15 dB PSNR, 0.016 D-SSIM, 0.049 LPIPS, 40 min, 90 MB (Table 3) | yes; per-scene values in the supplementary (Table 6) | 1352×1014, 300 frames, cloud from the first frame's SfM, 3 000 + 14 000 steps, per-scene batch from `arguments/dynerf/` |
| 4DGS native-4D (Yang et al.) | **yes** — 32.01 dB, 0.014 DSSIM, 0.055 LPIPS, 114 FPS (Table 1) | yes; average only | 300 frames, one held-out view, 30 000 steps, batch 4 (`configs/dynerf/*.yaml`) |
| Spacetime Gaussians (Li et al.) | **yes** — full 32.05 dB / 0.044 LPIPS, lite 31.59 dB / 0.047 LPIPS (Table 6) | yes | 1352×1014, first camera held out, 300 frames **trained as six 50-frame chunks**, 25 000-step snapshot, A6000 |
| Dynamic 3D Gaussians (Luiten et al.) | **no** — only Panoptic Sports in its own paper | no; 30.67 dB / 0.099 LPIPS **reported by the STG authors** (same Table 6) with their tuning | see decision 5 |
| Deformable-3DGS (Yang et al.) | not applicable (monocular) | — | — |

Two consequences:

* **Spacetime Gaussians is less affected by the 50-frame window than the others.** 50 frames is exactly its training unit, and our run is its first chunk. The paper's number is the average over six chunks, so it is still not identical, but it is the closest to like-for-like of the four.
* **For Dynamic 3D Gaussians the only external reference is someone else's tuning**, which is why decision 5 exists.

### Corrections this check produced

Both are recorded in [DISCLOSURES_MULTIVIEW.md](DISCLOSURES_MULTIVIEW.md) §8. The `ours_full` decoder is saved upstream (decision 3 rewritten). Notebook 05 overrode the official per-scene batch size of 4DGaussians (4 instead of 2 on four scenes); fixed.

### Sources

* Wu et al., *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering*, CVPR 2024 — [arXiv 2310.08528](https://arxiv.org/abs/2310.08528), Table 3 and supplementary Table 6; repository [`hustvl/4DGaussians`](https://github.com/hustvl/4DGaussians), `arguments/dynerf/`.
* Yang et al., *Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting*, ICLR 2024 — [arXiv 2310.10642](https://arxiv.org/abs/2310.10642), Table 1; repository [`fudan-zvg/4d-gaussian-splatting`](https://github.com/fudan-zvg/4d-gaussian-splatting), `configs/dynerf/`.
* Li et al., *Spacetime Gaussian Feature Splatting*, CVPR 2024 — [arXiv 2312.16812](https://arxiv.org/abs/2312.16812), Appendix B Table 6 and Appendix E.1; repository [`oppo-us-research/SpacetimeGaussians`](https://github.com/oppo-us-research/SpacetimeGaussians), `configs/n3d_*`, README, issues [#81](https://github.com/oppo-us-research/SpacetimeGaussians/issues/81) and [#119](https://github.com/oppo-us-research/SpacetimeGaussians/issues/119).
* Luiten et al., *Dynamic 3D Gaussians*, 3DV 2024 — [arXiv 2308.09713](https://arxiv.org/abs/2308.09713); repository [`JonathonLuiten/Dynamic3DGaussians`](https://github.com/JonathonLuiten/Dynamic3DGaussians), README and issues [#17](https://github.com/JonathonLuiten/Dynamic3DGaussians/issues/17), [#18](https://github.com/JonathonLuiten/Dynamic3DGaussians/issues/18).
* Gao et al., *HiCoM*, NeurIPS 2024 — [arXiv 2411.07541](https://arxiv.org/abs/2411.07541), Table 6 (quotes the Dynamic 3D Gaussians N3DV values from the STG paper).

## How to record the decision

When one is taken, do all three of:

1. set the knob in cell 0.1 of the affected notebook, or pass it on the command line as
   `--set NAME=VALUE` to `scripts/run_benchmark.py` (see the [running notes in the README](../README.md#running-on-a-cloud-gpu-over-ssh));
2. note the choice and the date in this file, replacing the entry's "not yet decided";
3. if the choice differs from what [METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md) §4 and §6 describe, update those sections too — they state the current defaults as fact.

Choices 1 to 3 are already recorded per run in the benchmark JSON (`budget_mode`, `num_frames`, `stg_model`), so a run always carries the decision it was made under, whatever this file says. **Choice 4 is not**: nothing in the JSON records which GPU produced a run, so it has to be written down here and repeated wherever the numbers are quoted.
