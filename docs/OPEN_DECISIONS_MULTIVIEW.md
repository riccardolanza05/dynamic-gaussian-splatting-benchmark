# Open decisions, multi-view study

The multi-view extension rests on a number of **judgement calls, not facts**. This page lists them: what was decided and why, what is still open, and what each alternative would cost. Each entry says what is implemented today, so that a decision is taken with the trade-off in view and not discovered when the tables are already built.

The decisions keep their original numbers, because the code and the other documents refer to them by number. Three were taken on 2026-10-02, two lapsed on the same day because the method they concerned was set aside, and four are open.

| # | Decision | Status | Implemented today |
|---|---|---|---|
| [1](#1-iteration-budget-of-dynamic-3d-gaussians) | Protocol A budget of Dynamic 3D Gaussians | **lapsed** (method set aside, 2026-10-02) | — |
| [2](#2-temporal-window) | Temporal window | **decided 2026-10-02**: 300 frames is the main study, 50 frames the secondary one | `NUM_FRAMES = 300` by default, `--set NUM_FRAMES=50` for the short window |
| [3](#3-which-spacetime-gaussians-model) | Spacetime Gaussians variant | **decided 2026-10-02**: both, as two rows; `ours_lite` alone if both cost too much. The threshold is open | `ours_lite` by default, `--set STG_MODEL=ours_full` for the second row |
| [4](#4-which-gpu) | Which GPU the whole study runs on | **open** — must be chosen before the first run | nothing: it is chosen when the machine is started |
| [5](#5-how-to-configure-dynamic-3d-gaussians-on-n3dv) | Configuration of Dynamic 3D Gaussians on N3DV | **lapsed** (method set aside, 2026-10-02) | — |
| [6](#6-the-protocol-a-budget) | Protocol A budget | **decided 2026-10-02**: each method at its official N3DV budget | 3 000 + 14 000 for 4DGaussians, 30 000 for the other two |
| [7](#7-protocol-b-on-the-300-frame-window) | Protocol B on the 300-frame window | **open** — to be taken after the Protocol A runs | Protocol B runs on the 50-frame window only |
| [8](#8-a-temporally-subsampled-window) | A temporally subsampled window (one frame in six) | **open** — to be taken after the smoke test | not implemented |
| [9](#9-lpips-backend) | LPIPS backend, now that its original reason is gone | **open**, low stakes | the pip `lpips` package in all three notebooks |

**Irreversible ones.** Changing the GPU (4) or the main window (2) after the runs means rerunning everything. The others are a knob, one notebook to rerun, extra runs in separate folders, or an analysis step.

---

## Refresher: what is measured, and how the three methods differ

### The two protocols

Every method is scored on two axes: **quality** (PSNR, SSIM, LPIPS and L1 on the held-out views) and **cost** (training time, peak VRAM, model size on disk).

* **Protocol A — official budget.** Every method trains for the number of optimisation steps its authors use on this dataset, and the quality it reaches is compared with what it cost. It answers *"run as its authors run it, how well does each method reconstruct, and at what cost?"*. Until 2026-10-02 Protocol A gave every method the same 30 000 steps, as in the monocular study; see decision 6.
* **Protocol B — fixed quality.** Every method trains until its test L1 drops below a per-scene target, and what it cost to get there is compared. It answers *"what does this quality cost?"*. The targets are derived from the Protocol A results, so Protocol B always runs after Protocol A.

### The dataset

**N3DV** (*Neural 3D Video*): about 20 fixed, synchronised cameras film a kitchen with a person moving in it. Each scene is 300 frames, 10 seconds at 30 fps. Camera `cam00` is held out for testing, as the dataset's own README states.

### The three methods, in one line each

| Method | How it represents motion | Consequence for this page |
|---|---|---|
| **4DGaussians (Wu)**, notebook 05 | One set of "canonical" 3D Gaussians plus a small network (HexPlane grid + MLP) that, given a time *t*, says how much to move, rotate and scale each Gaussian. | **One model for the whole sequence.** |
| **4DGS native-4D (Fudan)**, notebook 06 | The Gaussians are four-dimensional: they have an extent in time too. Rendering instant *t* means "slicing" each 4D Gaussian into a 3D one. | **One model for the whole sequence.** |
| **Spacetime Gaussians (Li)**, notebook 07 | Each Gaussian has an opacity that rises and fades over time and a polynomial trajectory. | One model, but **by construction it covers a block of 50 frames**: a 300-frame scene is six models. |

A fourth method, **Dynamic 3D Gaussians (Luiten)**, was part of the study until 2026-10-02. Why it was set aside is in [its own section](#dynamic-3d-gaussians-set-aside-2026-10-02).

---

## 2. Temporal window

**Where:** cell 0.1 of the three notebooks, the variable `NUM_FRAMES`.

**Decided 2026-10-02: the main study uses all 300 frames; a secondary study uses the first 50.**

### What the window is

Each scene is a 10-second video at 30 fps, 300 frames. The *window* is how many consecutive frames each method reconstructs. The test is always `cam00` inside the window: 300 test images, or 50.

### Why 300 is now the main window

Until 2026-10-02 the main window was 50 frames, for one reason: Dynamic 3D Gaussians costs a fixed number of steps *per frame* (108 000 steps at 50 frames, 608 000 at 300), so 50 was the longest window in which four methods fitted on one rented GPU. With that method set aside, the constraint is gone, and all three remaining methods run on 300 frames, which is what their papers report:

| Method | Published N3DV result, 300 frames, 1352×1014 | Source |
|---|---|---|
| 4DGaussians | 31.15 dB PSNR, 0.049 LPIPS | its paper, Table 3; per scene in its appendix, Table 6 |
| 4DGS native-4D | 32.01 dB PSNR, 0.055 LPIPS | its paper, Table 1 (average only) |
| Spacetime Gaussians, full | 32.05 dB PSNR, 0.044 LPIPS | its paper, Appendix B, Table 6 |
| Spacetime Gaussians, lite | 31.59 dB PSNR, 0.047 LPIPS | same table |

**What 300 frames buys.** The comparison with the papers, which the 50-frame window did not allow. In the monocular study that comparison was the quality check ([METHODOLOGY.md](METHODOLOGY.md) §7): it is what revealed the white/black background issue. An error common to all methods — poses converted wrongly, a wrong test split, a different resolution — moves every method by a similar amount, leaves the ranking plausible, and is visible only against an external number.

**What 300 frames costs.**

* **4DGaussians and 4DGS native-4D**: the same number of steps as at 50 frames, since each builds one model for the window. What grows is the data: six times the frames to extract and to load, and each evaluation renders 300 test views instead of 50.
* **Spacetime Gaussians**: six times the training (six independent 50-frame models per scene, as in its paper) and 300 per-frame COLMAP reconstructions per scene instead of 50. It is the method that dominates the budget; see decision 3.
* **Disk**: estimated 13–18 GB of extracted frames per scene (2–3 MB per PNG, about 20 cameras, 300 frames), against 2–3 GB at 50 frames. An estimate; the runs record the real figure (`scene_data_mb`).

### What the 50-frame study is for

It is kept as the secondary study, with the same three methods, for two reasons:

* **Protocol B runs there.** At 300 frames Protocol B is not defined for Spacetime Gaussians, which is six separate models (decision 7).
* **It is cheap**, and it is the first thing a smoke test exercises.

Its limits: no published number is comparable with it, and the first 1.67 seconds of a scene may not be representative of the whole (the flame of `flame_salmon_1`, for instance). Decision 8 is a proposal that addresses the second point.

### Spacetime Gaussians at 300 frames

Six independent models of 50 frames, one per block, which is how the paper covers a 300-frame sequence (each model is trained on a 50-frame sequence and the models are arranged in series). The merged result is defined as:

* **quality** (PSNR, SSIM, LPIPS, L1): the mean over the 300 test images;
* **training time, steps, images seen, storage**: summed over the six;
* **peak VRAM**: the maximum of the six.

### How it is implemented

* Notebooks 05 and 06 take the window from `NUM_FRAMES`; runs go to `<scene>_f300_…` or `<scene>_f50_…` folders.
* Notebook 07 trains in blocks whenever `NUM_FRAMES` is above 50, keeps each block's JSON under `<run>/blocks/block_<k>/` and merges them into one benchmark JSON. An interrupted scene resumes at the first unfinished block.
* The analysis keeps the two windows apart: `GS_STUDY=n3dv` reads the 300-frame runs and writes `results/n3dv/analysis/`; `GS_STUDY=n3dv_f50` reads the 50-frame runs and writes `results/n3dv/analysis_f50/`.

---

## 3. Which Spacetime Gaussians model

**Where:** cell 0.1 of notebook 07, the variable `STG_MODEL`.

**Decided 2026-10-02: both released variants are benchmarked, as two rows of the results. If running both costs too much, only `ours_lite` is run.** What "too much" means is open and waits for the smoke test.

### The two released models

The authors release two, with separate configs (`configs/n3d_lite/`, `configs/n3d_full/`) and separate rasterizers. They differ in how the Gaussians carry colour:

* **`ours_lite`** — each Gaussian holds an RGB colour directly.
* **`ours_full`** — each Gaussian holds a 9-number feature; the rasterizer produces a feature image and a small neural network (an MLP, `rgbdecoder`) turns it into RGB, using the view direction and the time.

In the paper's own table (Appendix B, Table 6) the full model is 0.46 dB better in PSNR (32.05 against 31.59), slightly better in LPIPS (0.044 against 0.047), twice as large (200 MB against 103 MB) and half as fast to render (140 FPS against 310).

### Why both

* **`ours_full` is the method as its authors present it**: the headline row of the paper.
* **`ours_lite` is the homogeneous comparison**: neither of the other two methods has a neural colour decoder. The authors themselves say the MLP "mainly compensates for view-dependent appearance changes" and that even subtle colour mismatches move the PSNR (issue [#119](https://github.com/oppo-us-research/SpacetimeGaussians/issues/119)).
* With both rows, a reader can see how much of the method's result is the spacetime representation and how much is the decoder.

### What it costs

One more full loop of notebook 07, the most expensive notebook. Two details:

* **The per-frame COLMAP stage is shared between the variants only if the scene data is kept.** By default a finished scene frees its frames and its COLMAP models (`DELETE_FRAMES_AFTER_TRAIN = True`), so the second variant would rebuild all of them. To reuse them, run the first variant with `--set DELETE_FRAMES_AFTER_TRAIN=false`, disk permitting. An earlier version of this page said the stage was "cached and shared" without this condition.
* **The only published timing** is "The training time for a 50-frame sequence is 40-60 minutes on a single NVIDIA A6000 GPU" (the paper, implementation details), without saying for which variant. At 300 frames a variant is 6 blocks × 6 scenes = 36 models, i.e. 24–36 hours on that card. For comparison, 4DGaussians reports 40 minutes per scene (Table 3, on an RTX 3090): about 4 hours for the six scenes.

### The fallback

If both are too expensive, `ours_lite` alone. The comparison with the paper survives, because the paper has a row for it (31.59 dB). What is lost is the paper's headline number.

**Open sub-point (3a): the threshold.** No number has been fixed for "too expensive". The smoke test gives the hours per variant on the chosen GPU ([how](#the-smoke-test)). An intermediate option exists: both variants on the 50-frame study, one on the 300-frame one.

### How it is implemented

`STG_MODEL` defaults to `ours_lite`. `ours_full` runs go to their own folders (`<scene>_f300_iters30000_full`), `stg_model` is recorded in every JSON, and the analysis reports the two variants as two methods; a variant that was not run is left out of the figures.

---

## 4. Which GPU

**Where:** not in the code — it is what you select when you start the machine. See also the [running notes in the README](../README.md#running-on-a-cloud-gpu-over-ssh).

**Open. It has to be chosen before the first run.**

### Why this is a decision and not a detail

Two of the monitored metrics, **training time** and **peak VRAM**, are not properties of the method but of the pair *method + card*. Therefore:

* **every run must use the same GPU type**, including any rerun of a single scene;
* the benchmark JSON **does not record** which card produced a number, so it has to be written down in the documents;
* **one run at a time** on the GPU, because VRAM is measured on the whole device (by polling `nvidia-smi`).

> If two methods are trained on two different GPUs, their training times cannot be compared, and the "quality per unit of cost" question that Protocol B exists to answer has no answer. **Mixing GPU types means rerunning.**

### The options

| GPU | Pros | Cons |
|---|---|---|
| **T4** (16 GB) | The card of the monocular study. Cheapest per hour. | The slowest. Spacetime Gaussians needs `--gtisint8 1` to fit (below). The "continuity" with the monocular study is partly illusory: dataset, resolution and window differ, and so does the machine around the GPU. |
| **L4** (24 GB) | Good speed/price balance; fits Spacetime Gaussians as its README requires. | Times are not comparable with the monocular study. |
| **A10G** (24 GB) | Faster still. | More expensive per hour; same incomparability. |
| **A100** (40/80 GB) | The fastest. | Pays for memory that may not be needed: in the monocular study no run exceeded 5.1 GB of VRAM. |

**Memory.** Spacetime Gaussians' README states: "You need 24GB GPU memory to train on the Neural 3D Dataset", because "training images are loaded into GPU memory". Its code has an option that holds them as 8-bit integers instead of floats (`gtisint8`, `thirdparty/gaussian_splatting/arguments/__init__.py:147`; it is not mentioned in the README), which is lossless for 8-bit PNG frames. On a 16 GB card it is required (`--set EXTRA_TRAIN_ARGS="--gtisint8 1"`); on a 24 GB card it should not be. Whether 24 GB is enough for `ours_full`, and for the other two methods at 300 frames, is to be measured.

### What can honestly be said about cost today

No estimate of GPU hours is given here: nothing has run yet, and the published figures come from other cards.

| Method | Published training time on N3DV | Conditions |
|---|---|---|
| 4DGaussians | 40 min per scene | 300 frames, 3 000 + 14 000 steps, RTX 3090 (paper, Table 3 and §5.1) |
| Spacetime Gaussians | 40–60 min per 50-frame model | NVIDIA A6000 (paper, implementation details); variant not stated |
| 4DGS native-4D | not reported | — |

### The smoke test

A smoke test is an ordinary Protocol A run of one scene, stopped after a few thousand steps and written to its own folder so that it can never be read as a result:

```bash
python3 scripts/run_benchmark.py notebooks/05_4dgaussians_wu_n3dv.ipynb \
    --set RUN_MODE=single --set SCENE=sear_steak --set NUM_FRAMES=50 \
    --set MAX_ITERATIONS=2000 --set EVAL_EVERY_N_ITERS=500 \
    --set ROOT_NAME=dgs-smoke --set STORAGE_MODE=local
```

What to run, on the GPU type being considered:

1. the three notebooks at `NUM_FRAMES=50`, notebook 07 once per variant;
2. notebooks 05 and 06 again at `NUM_FRAMES=300`: their step count does not change with the window, but data loading and the evaluation of 300 test views do, and neither extrapolates from 50 frames;
3. Spacetime Gaussians at 300 frames does **not** need its own smoke run: it is six times the 50-frame one, COLMAP included.

Then `python3 scripts/estimate_gpu_hours.py <workdir>/bench_out/dgs-smoke` prints what was measured (seconds per step, seconds per evaluation, peak VRAM, preparation time, disk per scene) and the hours the full study would take, per method and window.

**How far to trust it.** The training hours are a lower bound: a short run stops while the number of Gaussians is still growing (densification runs until step 9 000 for Spacetime Gaussians, 10 000 for 4DGaussians, 15 000 for 4DGS native-4D), and a step gets slower as they grow. A single complete run of the cheapest method is the way to see by how much.

**Recommendation:** a 24 GB card, stated wherever the results are quoted, confirmed by the smoke test and by the budget available.

---

## 6. The Protocol A budget

**Where:** `MAX_ITERATIONS` in cell 0.1 of the three notebooks.

**Decided 2026-10-02: each method runs for the number of steps its authors use on N3DV. There is no longer a step count common to the methods.**

### The official budgets (re-read in the repositories on 2026-10-02)

| Method | Official N3DV schedule | Batch (views per step) | Images seen | Densification stops at |
|---|---|---|---|---|
| 4DGaussians | 3 000 coarse + **14 000** fine, on every scene (`arguments/dynerf/default.py`; the per-scene files change only the batch) | 4 on `coffee_martini` and `flame_salmon_1`, 2 on the other four | 68 000 or 34 000 | 10 000 |
| 4DGS native-4D | **30 000**, on every scene (`configs/dynerf/<scene>.yaml`) | 4 | 120 000 | 15 000 |
| Spacetime Gaussians | **30 000** (the default of `arguments/__init__.py`; the N3DV configs do not set it) | 2 | 60 000 per model, 360 000 for the six models of a scene | 9 000 |

### Why

The monocular study gave every method 30 000 steps, and its own results showed that a step is not a common unit of work: the methods use different batch sizes, so the same number of steps is a very different amount of training ([METHODOLOGY.md](METHODOLOGY.md) §3.5, [RESULTS.md](RESULTS.md) §4). On N3DV a uniform 30 000 would also run 4DGaussians at nearly twice its official budget (17 000). Running each method as its authors do gives the number its authors would report, which is what the comparison with the papers needs.

### What is given up

* **The "at equal steps" table.** The step column now differs: 17 000, 30 000, 30 000. The comparison across methods is on images seen and on time, which the analysis already uses as budget axes. The images seen differ too, by up to a factor of 3.5 (34 000 against 120 000): that is the methods' own tuning, and it is declared, not removed.
* **It can be recovered for 4DGaussians alone**, by rerunning notebook 05 with `--set MAX_ITERATIONS=27000`. Its learning-rate schedules are fixed in steps (`position_lr_max_steps = 20 000`, `arguments/__init__.py:119`; `scene/gaussian_model.py:185–196`), so the longer run passes through the same states as the official one up to step 14 000.

### Spacetime Gaussians: two readings of one run

The official flow trains 30 000 steps and then evaluates **one snapshot per scene**, set by `test_iteration` in the scene's config. It is **not the same on every scene**:

| Scene | `test_iteration` (both `n3d_lite` and `n3d_full`) |
|---|---|
| `cook_spinach`, `cut_roasted_beef`, `flame_steak`, `sear_steak` | 25 000 |
| `coffee_martini` | 10 000 |
| `flame_salmon_1` | 12 000 |

An earlier version of this page said 25 000 for all six. The two scenes with an earlier snapshot are also the two whose configs set `densify: 2`.

**Decided:** train 30 000 and report both readings — the end of the run, and the official snapshot of each scene, which is the one to set next to the paper. The snapshot costs nothing: all three values fall on the sampling grid, and each run records its own (`official_test_iteration`). The analysis writes both (`table_official_readout.csv`).

### Sampling: about 30 points per run

With budgets that differ, sampling every 1 000 steps would give the methods curves of different resolution. Each notebook now samples so that a run gives about 30 points:

| Method | Interval | Samples |
|---|---|---|
| 4DGaussians | every 500 fine steps | 28, plus the baseline one (the coarse stage is not sampled, as in the monocular study) |
| 4DGS native-4D | every 1 000 steps | 30, plus the baseline one |
| Spacetime Gaussians | every 1 000 steps | 30, plus the baseline one, per model |

Each sample evaluates the whole test split, 300 views in the main study. That time is excluded from the training time, but it is GPU time that is paid for; the smoke test measures it.

### What this means for Protocol B

**The target calibration is unchanged.** The rule is `target(scene) = 1.05 × max over methods (minimum eval L1 on that method's Protocol A curve)` ([PROTOCOL_B_CALIBRATION.md](PROTOCOL_B_CALIBRATION.md)). It never assumed equal budgets; it needs every method to have actually reached the minimum it is credited with, which holds for a curve of any length. The bar is now "the worst of the methods' bests, as their authors run them".

**For Spacetime Gaussians the minimum is taken over the whole 30 000-step curve**, not over the curve cut at the official snapshot. On the two scenes with an early snapshot the two can differ; taking the whole curve is the rule already applied to the other methods ("the minimum of the curve"), and it can be revisited when the curves exist.

**Protocol B itself does not change.** It stops when the target is reached, its safety cap (60 000 steps) is above every official budget, and the schedules of the three methods are fixed in steps, so a Protocol B run follows the same trajectory as the Protocol A run until it stops.

---

## 7. Protocol B on the 300-frame window

**Where:** notebook 07, which refuses a Protocol B run in blocks; the Protocol B targets in cell 0.1 of the three notebooks.

**Open. To be taken after the Protocol A runs, when the curves show how much it matters.**

### The problem

Protocol B stops a run "the first time the target is reached". At 300 frames Spacetime Gaussians is six independent models, each trained on its own 50 frames: there is no single run whose first crossing can be read. Today Protocol B is therefore run on the 50-frame window only, where the method is one model.

### Option A — Protocol B on the 50-frame window only (implemented)

* **Pros:** defined in the same way for all three methods; nothing to add.
* **Cons:** the main study has no Protocol B. The 50-frame targets say nothing about the full sequence.

### Option B — at 300 frames, with Spacetime Gaussians judged on its worst block

Proposed by the study owner (2026-10-02). The target of a scene is calibrated using, for Spacetime Gaussians, the **worst of its six models** (the largest of the six curve minima), so that every block can reach it. Each block then runs to the target.

* **Pros:** Protocol B on the main window, for all three methods.
* **What has to be defined, and is not yet:**
  * **the cost of the scene.** Six models are needed to render the sequence, so the natural definition is the *sum* of the six first-crossing times (and of the six storages), with VRAM the maximum — the same rule as the Protocol A merge;
  * **what "reaching the target" means for the other two methods.** Their L1 is a mean over 300 views. Spacetime Gaussians would have to reach the target on *each* 50-view block, which is a stricter condition than reaching it on average: a method that is good on five blocks and slow on one is charged for the slow one.
* **The hope behind the proposal** is that the six models converge quickly and that, on every scene, another method is the one that sets the bar, so that the asymmetry never binds. That can be checked on the Protocol A curves before any Protocol B run: if Spacetime Gaussians' worst block is never the binding minimum, the problem does not arise.
* **Cost:** new code in notebook 07 (Protocol B per block, and the merge of six first crossings), and a Protocol B loop at 300 frames for notebooks 05 and 06.

### Option C — no Protocol B run at 300 frames: read the crossing on the Protocol A curves

The Protocol A curves already contain "the first sample at which the target is met". For Spacetime Gaussians the merged curve is the mean over the six blocks at equal steps.

* **Pros:** no GPU time and no new training code; the same rule for all three.
* **Cons:** the resolution is the Protocol A grid (500 or 1 000 steps), far coarser than the 30-second sampling of a real Protocol B run, and for Spacetime Gaussians it assumes the six blocks are stopped at the same step.

**Current default:** A. **Recommendation:** look at the 300-frame Protocol A curves first; C is the cheap way to see whether B is worth its code.

---

## 8. A temporally subsampled window

**Where:** nowhere yet. It would be a new knob in cell 0.1 of the three notebooks.

**Open. Proposed on 2026-10-02 by the project's collaborators; to be taken after the smoke test.**

### The idea

Instead of the first 50 consecutive frames, take **one frame in six**: frames 0, 6, 12, … 294. That is still 50 frames, so it is still one Spacetime Gaussians block and costs what the 50-frame study costs, but it spans the whole 10 seconds, at 5 fps instead of 30. It asks how the methods cope with a lower temporal resolution.

### What it would and would not solve

* **It makes the short window representative of the whole scene.** The first 50 frames are 1.67 seconds; one frame in six covers events anywhere in the sequence.
* **It is not needed to cover the whole sequence**: the main study already does, with all 300 frames, and Spacetime Gaussians is not limited to 50 frames (it uses six models). The subsampled window is a different experiment, not a substitute for the main one.
* **Its results are not comparable with any paper.**

### The risk is temporal, not spatial

Every instant keeps all its cameras, so the multi-view (spatial) consistency of each frame is untouched. What changes is the motion between consecutive training frames, which is six times larger:

* **4DGaussians** encodes time in a grid whose official N3DV resolution is 150 steps for 300 frames (`resolution: [64, 64, 64, 150]` in `arguments/dynerf/default.py`). Fifty samples over the same span leave most of that grid unobserved.
* **4DGS native-4D** would see the same 10-second duration with six times fewer observations per unit of time.
* **Spacetime Gaussians** models each Gaussian's motion with a polynomial over one block. A block would then span 10 seconds instead of 1.67. Its authors did try a long span once: on `flame_salmon` they trained a single model on all 300 frames and report 29.17 dB PSNR and 0.068 LPIPS, against 29.48 dB and 0.063 for six 50-frame models, with a smaller total size (216 MB against 300 MB) and less training time per frame (the paper's supplementary, section *Longer Video Sequence*). That model saw all 300 frames, though, not one in six.

Whether the methods degrade gracefully is exactly what the experiment would measure. A large loss of quality is a possible and legitimate outcome.

### What it needs

A `FRAME_STRIDE` knob, and four places that honour it: the frame extraction, and the three per-method preparations (the reader constants patched in notebook 05, the frame times written by notebook 06, the per-frame COLMAP folders and start frame of notebook 07). Run folders and JSON must carry the stride, as they carry the window, and the analysis needs a study for it. Roughly a day of work including the checks; in GPU time, one more loop of each notebook at the cost of the 50-frame study.

A possible extension, not costed: the frames that were skipped are available as a test of **temporal interpolation** (rendering instants the model never saw). All three methods are continuous in time, so it is possible in principle; it needs more code than the stride itself.

### Options

* **A — do not run it** (today's state).
* **B — add it as a third study**, next to the 300-frame and the contiguous 50-frame ones.
* **C — let it replace the contiguous 50-frame study.** Protocol B would then run on a window that is representative of the whole scene, at the same cost.

**Current default:** A. **Recommendation:** decide once the smoke test has given the cost of a 50-frame loop; if the experiment is run, C is the option that adds no GPU time.

---

## 9. LPIPS backend

**Where:** `LPIPS_BACKEND` in cell 0.1 of the three notebooks.

**Open, low stakes.**

The multi-view notebooks force the pip `lpips` package for every method. The reason was Dynamic 3D Gaussians, which does not bundle `lpipsPyTorch` as the other repositories do: leaving the choice to each repository would have put one method on a different implementation. With that method set aside, the reason no longer applies.

* **Option A — keep the pip package (implemented).** One implementation by construction, whatever each repository bundles. LPIPS is then not comparable with the monocular study, which used the bundled package.
* **Option B — use each repository's bundled `lpipsPyTorch`, as the monocular study does.** Continuity between the two studies. It requires checking that all three repositories bundle it and that the three copies are the same code; that check has not been made.

Both are LPIPS with the VGG backbone and the official linear weights, so the difference is expected to be small. **Recommendation:** keep A unless LPIPS has to be compared across the two studies.

---

## 1. Iteration budget of Dynamic 3D Gaussians

**Lapsed on 2026-10-02**, with the method. The question was whether to run it at its authors' schedule (10 000 steps on the first frame + 2 000 per later frame, 108 000 at 50 frames) or forced to the 30 000 steps of the others (about 408 per frame). The full write-up is in the history of this file (commit `39f6069` and earlier).

## 5. How to configure Dynamic 3D Gaussians on N3DV

**Lapsed on 2026-10-02**, with the method. The question was whether to give it segmentation masks estimated by this project, or the recipe the Spacetime Gaussians authors used for the only published N3DV result of that method (no masks, no floor loss, a higher learning rate for the scales; issue [#81](https://github.com/oppo-us-research/SpacetimeGaussians/issues/81) of their repository). The full write-up is in the history of this file.

---

## Dynamic 3D Gaussians: set aside (2026-10-02)

**Decided 2026-10-02: the method is set aside for now.** Notebook 04 and its entries in the analysis are no longer part of the study. Nothing about the method was measured by this project: the decision rests on published numbers and on what including it would have required.

### What the published numbers say

The only published result of Dynamic 3D Gaussians on N3DV was made by the Spacetime Gaussians authors, on their own protocol: 300 frames, 1352×1014, the first camera held out (their paper, Appendix B, Table 6; its caption says that FPS is measured at 1352×1014 and that the size is the total for 300 frames).

| Method | PSNR ↑ | LPIPS ↓ | Model size | FPS ↑ |
|---|---|---|---|---|
| Dynamic 3D Gaussians | 30.67 | 0.099 | 2772 MB | 460 |
| Spacetime Gaussians, lite | 31.59 | 0.047 | 103 MB | 310 |
| Spacetime Gaussians, full | 32.05 | 0.044 | 200 MB | 140 |

For context, the other two methods' own papers report, at the same resolution: 4DGaussians 31.15 dB, 0.049 LPIPS, 90 MB, 30 FPS (Table 3); 4DGS native-4D 32.01 dB, 0.055 LPIPS, 114 FPS (Table 1). These come from different papers and different GPUs, so the comparison is indicative.

* **Last in quality on both metrics.** 0.5–1.4 dB behind in PSNR, and about twice as bad in LPIPS (0.099 against 0.044–0.055), the widest gap in the table.
* **Model size out of scale.** 2.7 GB against 90–200 MB, 14–30 times larger, because it stores the position and rotation of every Gaussian for every frame.
* **The only column it wins is rendering speed.**
* **That number already comes from a tuned configuration.** The Spacetime Gaussians authors write that with the default hyper-parameters the rendering quality was "subpar", and that they tuned them for this dataset.
* **Uneven across scenes.** The per-scene values of the same runs, given in the Spacetime Gaussians paper's per-scene table and quoted by HiCoM (Table 6), are 32.97–33.68 dB on `cook_spinach`, `flame_steak` and `sear_steak`, but 26.49 on `coffee_martini` and 26.92 on `flame_salmon`.

### Why the rendering speed does not outweigh the rest

The judgement of the project: for its purpose, a fluid playback, a rendering speed above roughly 120 FPS is already more than enough, and the difference between 140 and 460 FPS is not perceptible on an ordinary display. The one advantage of the method is therefore not worth a clear loss of quality and a model more than ten times larger.

Two limits of this argument, stated so that it is not read as more than it is:

* **Rendering speed is not measured by this benchmark.** The monitor records quality, training time, steps, images seen, number of Gaussians, peak VRAM and model storage. The FPS figures above are the papers', each measured on its authors' hardware.
* **It is an argument about this method's advantage, not a claim that the remaining methods are all above 120 FPS.** 4DGaussians reports 30 FPS on N3DV in its own paper.

### What including it would have required

It was the method behind decisions 1 and 5, and the reason the main window could not be longer than 50 frames:

* 108 000 steps at 50 frames, 608 000 at 300;
* no official N3DV configuration, and no data-preparation code released by its authors;
* foreground/background masks that N3DV does not provide, to be either estimated by this project or replaced by another group's recipe;
* a Protocol B that would not have been comparable with the other methods, because stopping early truncates the sequence instead of producing a less converged model;
* a data conversion that was entirely this project's code and had never run on a GPU.

The title of the paper is *"Tracking by Persistent Dynamic View Synthesis"*: its goal is dense, physically consistent 3D tracking, and it accepts constraints that cost image quality to get it. This benchmark measures novel-view synthesis, so the method's strength would not have been measured, while its costs would.

### What "set aside" means

* The main comparison and the short-window one have the same three methods.
* Decisions 1 and 5 lapse; the 50-frame constraint on the main window disappears (decision 2).
* Notebook 04 is removed from the repository. It was never run; its text and code remain in the history of the branch.
* The method stays among the five studied by the project ([METHODOLOGY.md](METHODOLOGY.md) §1, [REFERENCES.md](REFERENCES.md)).

**What the data cannot rule out:** with a different configuration, or on a short window, it might come closer to the others on the simpler scenes. No published number suggests it would overtake them in quality.

---

## How the decisions interact

* **3 and 4 are tied by the budget.** Spacetime Gaussians at 300 frames is 36 models per variant; whether both variants are affordable is what the smoke test on the chosen GPU has to say.
* **2 and 7.** Making 300 frames the main window leaves the main study without Protocol B until decision 7 is taken.
* **2 and 8.** The subsampled window (8) is a candidate replacement for the contiguous 50-frame study, not for the main one.
* **6 and 7.** With official budgets, the Protocol B targets are "the worst of the methods' bests as their authors run them"; the calibration needs no change.

## What the papers and repositories say

Checked on 2026-09-29 and re-read in full on 2026-10-02, against the papers and the current default branch of each repository.

| Method | Official N3DV result | Settings behind the number |
|---|---|---|
| 4DGaussians (Wu et al.) | 31.15 dB PSNR, 0.016 D-SSIM, 0.049 LPIPS, 40 min, 30 FPS, 90 MB (Table 3); per scene in the appendix (Table 6) | 1352×1014, 300 frames, RTX 3090, 3 000 + 14 000 steps |
| 4DGS native-4D (Yang et al.) | 32.01 dB, 0.014 DSSIM, 0.055 LPIPS, 114 FPS (Table 1); average only | 300 frames, one view held out, 30 000 steps, batch 4, densification stopped at the midpoint |
| Spacetime Gaussians (Li et al.) | full 32.05 dB / 0.044 LPIPS / 200 MB / 140 FPS; lite 31.59 dB / 0.047 LPIPS / 103 MB / 310 FPS (Appendix B, Table 6) | 1352×1014, first camera held out, 300 frames as six 50-frame models, 40–60 min per model on an A6000 |
| Deformable-3DGS (Yang et al.) | not applicable: monocular only | — |

### Corrections produced by the full re-read of 2026-10-02

Recorded in [DISCLOSURES_MULTIVIEW.md](DISCLOSURES_MULTIVIEW.md) §7. None affects a result, since no run has been made.

* **Spacetime Gaussians' test snapshot is per scene**, not 25 000 everywhere (decision 6).
* **4DGS native-4D at 300 frames now uses the official `time_duration: [0, 10]`.** The notebook wrote `[0, (N−1)/30]`, i.e. `[0, 9.967]` at 300 frames. It now writes `[0, N/30]`.
* **The GPU of the 4DGaussians paper is stated**: a single RTX 3090 (§5.1). An earlier version of this page said it was not.
* **The 4DGaussians paper and its repository disagree on two settings.** The paper's appendix (A.1) says "The batch size in training is set to 1" and that the dense point cloud is downsampled "lower than 100k"; the repository's N3DV configs use batch 4 or 2, and its README says the cloud is downsampled "to less than 40000 points". This study follows the repository, which is what produces the runs.
* **The per-frame COLMAP stage of Spacetime Gaussians is not shared between its two variants by default** (decision 3).

Everything else on this page that cites a table, a file or an issue was found as cited, including the line numbers. The quoted sentences were checked on the README files, on the issue threads and on the LaTeX sources of the papers, not on rendered pages.

### Sources

* Wu et al., *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering*, CVPR 2024 — [arXiv 2310.08528](https://arxiv.org/abs/2310.08528), Table 3, §5.1, appendix A.1 and Table 6; repository [`hustvl/4DGaussians`](https://github.com/hustvl/4DGaussians), `arguments/dynerf/`, `arguments/__init__.py`, `scene/gaussian_model.py`, `scripts/downsample_point.py`, README.
* Yang et al., *Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting*, ICLR 2024 — [arXiv 2310.10642](https://arxiv.org/abs/2310.10642), Table 1; repository [`fudan-zvg/4d-gaussian-splatting`](https://github.com/fudan-zvg/4d-gaussian-splatting), `configs/dynerf/`, `scripts/n3v2blender.py`.
* Li et al., *Spacetime Gaussian Feature Splatting*, CVPR 2024 — [arXiv 2312.16812](https://arxiv.org/abs/2312.16812), Appendix B Table 6 (the per-scene table next to it has the same Dynamic 3D Gaussians runs), the implementation details (training time), and the supplementary section *Longer Video Sequence*; repository [`oppo-us-research/SpacetimeGaussians`](https://github.com/oppo-us-research/SpacetimeGaussians), `configs/n3d_lite/`, `configs/n3d_full/`, `thirdparty/gaussian_splatting/arguments/__init__.py`, README, issues [#81](https://github.com/oppo-us-research/SpacetimeGaussians/issues/81) and [#119](https://github.com/oppo-us-research/SpacetimeGaussians/issues/119).
* Luiten et al., *Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis*, 3DV 2024 — [arXiv 2308.09713](https://arxiv.org/abs/2308.09713); repository [`JonathonLuiten/Dynamic3DGaussians`](https://github.com/JonathonLuiten/Dynamic3DGaussians).
* Gao et al., *HiCoM*, NeurIPS 2024 — [arXiv 2411.07541](https://arxiv.org/abs/2411.07541), Table 6 (quotes the Dynamic 3D Gaussians N3DV values from the Spacetime Gaussians paper).
* Li et al., *Neural 3D Video Synthesis from Multi-View Video* — dataset [README](https://github.com/facebookresearch/Neural_3D_Video): "cam00.mp4 is the center reference camera which we held out for testing".

## How to record a decision

When one is taken, do all three of:

1. set the knob in cell 0.1 of the affected notebook, or pass it on the command line as `--set NAME=VALUE` to `scripts/run_benchmark.py` (see the [running notes in the README](../README.md#running-on-a-cloud-gpu-over-ssh));
2. note the choice and the date in this file, in the table at the top and in the entry;
3. update [METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md) if the choice changes what it states as fact.

The window, the Spacetime Gaussians variant and the budget are recorded per run in the benchmark JSON (`num_frames`, `stg_model`, `max_iterations`), so a run always carries the decisions it was made under. **The GPU is not**: nothing in the JSON records which card produced a run, so it has to be written down here and repeated wherever the numbers are quoted.
