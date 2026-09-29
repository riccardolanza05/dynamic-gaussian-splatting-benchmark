# Methodology, multi-view study (Neural 3D Video)

This document is the multi-view counterpart of [METHODOLOGY.md](METHODOLOGY.md). It describes how the benchmark was extended from the monocular D-NeRF setting to the multi-camera setting of *Neural 3D Video Synthesis from Multi-View Video* (N3DV), which methods it covers, what had to be aligned across four repositories instead of three, and — the part that took most of the work — **which differences between the methods could be removed and which could only be declared**.

The instrumentation, the metric definitions, the two protocols and the reporting are unchanged. Everything in [METHODOLOGY.md](METHODOLOGY.md) §2 (instrumentation), §4 (definition of the evaluation loss) and §6 (protocols) applies here verbatim and is not repeated.

Status: the notebooks and the protocol are in place; **no multi-view training run has been made yet**, so there are no results and the Protocol B targets are empty by construction. See §8.

Three companion pages: [RUNNING_ON_LIGHTNING.md](RUNNING_ON_LIGHTNING.md) is the step-by-step operating manual (get a GPU, reach it over SSH, send every result to one Google Drive folder, build the tables); [OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) collects the six choices that are still open (the budget and the configuration of the frame-by-frame method, the 50-frame window, which Spacetime Gaussians model, which GPU, and the Protocol A budget), each with what the alternative would cost; [DISCLOSURES_MULTIVIEW.md](DISCLOSURES_MULTIVIEW.md) collects what a reader of the results has to be careful about, and needs no decision.

## 1. Scope

**Goal.** The same controlled comparison, on the setting the monocular study could not reach: synchronised multi-camera video, where every instant is observed by many cameras at once.

**Methods benchmarked** (the four of the five studied that accept multi-camera input):

| Method | Repository | Representation | Notebook | Status before this extension |
|---|---|---|---|---|
| Dynamic 3D Gaussians (Luiten et al., 3DV 2024) | [`JonathonLuiten/Dynamic3DGaussians`](https://github.com/JonathonLuiten/Dynamic3DGaussians) | one set of Gaussians optimised frame by frame, with persistence losses | [`04`](../notebooks/04_dynamic3dgaussians_luiten_n3dv.ipynb) | studied, never benchmarked |
| 4DGaussians / HexPlane (Wu et al., CVPR 2024) | [`hustvl/4DGaussians`](https://github.com/hustvl/4DGaussians) | canonical space + factorised HexPlane grid + light decoder | [`05`](../notebooks/05_4dgaussians_wu_n3dv.ipynb) | benchmarked on monocular D-NeRF |
| 4DGS, native 4D primitives (Yang et al., ICLR 2024) | [`fudan-zvg/4d-gaussian-splatting`](https://github.com/fudan-zvg/4d-gaussian-splatting) | native 4D Gaussians with finite temporal extent | [`06`](../notebooks/06_4dgs_native4d_fudan_n3dv.ipynb) | benchmarked on monocular D-NeRF |
| Spacetime Gaussians (Li et al., CVPR 2024) | [`oppo-us-research/SpacetimeGaussians`](https://github.com/oppo-us-research/SpacetimeGaussians) | temporal radial-basis opacity + polynomial motion, per-frame COLMAP init | [`07`](../notebooks/07_spacetime_gaussians_li_n3dv.ipynb) | studied, never benchmarked |

Deformable-3DGS is left out: it targets single-camera scenes (NeRF-DS, HyperNeRF, D-NeRF) and has no multi-view data path.

**Dataset.** N3DV, six scenes (`coffee_martini`, `cook_spinach`, `cut_roasted_beef`, `flame_salmon_1`, `flame_steak`, `sear_steak`), 21 synchronised cameras (fewer where the release filtered out unsynchronised streams), 300 frames at 30 fps, native 2704×2028.

**Hardware.** A single cloud GPU, on a [lightning.ai](https://lightning.ai) Studio. The monocular study ran on a Google Colab Tesla T4; the multi-view runs are not tied to that machine, and the notebooks work unchanged on a Studio, on Colab or on any Linux machine with an NVIDIA card (see [RUNNING_ON_LIGHTNING.md](RUNNING_ON_LIGHTNING.md)).

**Which GPU is itself a decision**, and it has to be taken before the first run: training time and peak VRAM are two of the ten monitored metrics, and both are properties of the pair (method, GPU). Every run of the study must therefore use the same GPU type, and that type must be reported next to the numbers. See [OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) §4.

## 2. The three conventions that make the comparison possible

### 2.1 Resolution: 1352×1014

Half of the native 2704×2028. This is not a compromise: it is the resolution the **official N3DV configuration of every one of the four methods** uses. 4DGaussians' `Neural3D_NDC_Dataset` hard-codes `img_wh = (1352, 1014)`; fudan-zvg's `configs/dynerf/*.yaml` and Spacetime Gaussians' `configs/n3d_*/**.json` both set `resolution: 2` on the native video; Dynamic 3D Gaussians has no N3DV configuration and is given the same.

The frames are therefore extracted **once**, at 1352×1014, by a cell shared by the four notebooks, and each method is then told not to halve them again (`resolution: 1` for fudan-zvg and for Spacetime Gaussians, `--resolution 1` on the command line). The effective resolution is the official one; only the place where the downscaling happens moves. Every benchmark entry records `eval_resolution`, as in the monocular study, so the guarantee is permanent rather than a claim.

### 2.2 Test split: `cam00`, the whole window

The N3DV README states that `cam00` is the centre reference camera and is **held out for testing**. All three published methods follow it, and the fourth is given the same split. With the window of §2.3 the test set is exactly **50 views** — one per frame of the held-out camera — for every method, so `MAX_EVAL_VIEWS` stays at `0` (evaluate everything) exactly as in the monocular study and the evaluation L1 is directly comparable across methods.

### 2.3 Temporal window: the first 50 frames

This is the load-bearing decision of the extension, and the one that costs the most.

* **Spacetime Gaussians covers a sequence in chunks of 50 frames** by construction (`duration: 50` in every official N3DV config). A 300-frame run is six models, six trainings and six times the storage.
* **Dynamic 3D Gaussians costs a fixed number of optimiser steps per frame** — 10 000 for the first and 2 000 for each later one. Its total budget is linear in the length of the sequence: 108 000 steps for 50 frames, 608 000 for 300.
* **Disk.** One scene at 1352×1014, 21 cameras × 50 frames is already 15–20 GB of PNG; 300 frames would be six times that, on a runtime that also has to hold the models.

Fifty frames (1.67 s at 30 fps) is the longest window in which all four methods can be trained on the same hardware, and it is the window in which Spacetime Gaussians' native chunk equals the whole sequence, so no method is penalised by chunking. `NUM_FRAMES` is a configuration value in all four notebooks, it is written into **every** benchmark JSON (`num_frames`) and into every run folder name (`<scene>_f50_iters30000`), and `run_is_complete()` refuses to reuse a run made with a different window. Runs from different windows therefore cannot be mixed by accident.

**What this costs.** No published N3DV number is comparable with these runs: every paper reports the 300-frame sequence. The validation against the literature of [METHODOLOGY.md](METHODOLOGY.md) §7, which was an important check in the monocular study, **has no counterpart here**. The multi-view results are internally comparable and externally they are not.

## 3. Instrumentation: the same monitor, two new hook points

The monitor is the one described in [METHODOLOGY.md](METHODOLOGY.md) §2: a few lines appended to `train.py`, before `if __name__ == "__main__":`, that wrap a function the training loop already calls at every iteration. Upstream behaviour, hyper-parameters and command line are untouched, a pristine copy is kept as `train.py.orig`, and without `BENCH_CONFIG` the file behaves exactly as upstream.

Two of the four repositories expose the same `training_report(...)` as the monocular ones, and the monitor binds its arguments with `inspect.signature().bind()` as before. **The other two have no such function at all.**

| Repository | Wrapped function | Where the loop state comes from |
|---|---|---|
| hustvl/4DGaussians | `training_report(...)` | the signature |
| fudan-zvg/4d-gaussian-splatting | `training_report(...)` | the signature |
| oppo-us-research/SpacetimeGaussians | `controlgaussians(...)` (from `helper_train`) | signature + calling frame |
| JonathonLuiten/Dynamic3DGaussians | `report_progress(...)` | signature + calling frame |

Both substitutes are module-level names that the loop resolves at call time, so rebinding them in `train.py` works exactly as rebinding `training_report` does, and both are called once per iteration inside the loop's `torch.no_grad()` block. What their narrower signatures do not carry — the render function, the pipeline parameters, the background, the current loss, the frame index, the per-frame snapshots — is read from the **calling frame** with `sys._getframe`. The frame is only read, never written.

**The monitor core is byte-identical in the four notebooks.** It is delimited by `# === BENCHMARK MONITOR CORE ===` / `# === END BENCHMARK MONITOR CORE ===`, and everything repository-specific is in a glue block appended after it, behind ten named functions: `_stage_is_evaluable`, `_stage_of`, `_global_iteration`, `_num_gaussians`, `_train_batch_loss`, `_force_eval`, `_iter_eval_views`, `_render_pair`, `_save_model`, `_maybe_track_best`. A smoke test asserts the identity of the core across the four files, so "the same methodology" is checked rather than asserted.

Three fields were added to every entry: `num_frames`, `held_out_camera` and `dataset`, plus the per-method extras a method needs (`timesteps_done` for the frame-by-frame method). Nothing was removed.

## 4. Differences between the methods: removed, or declared

This is the section the extension exists for. A difference is **removed** when it can be equalised without changing what a method is, and **declared** when equalising it would report a number about a method nobody runs.

### 4.1 Removed

| # | Difference | How it was removed |
|---|---|---|
| 1 | **Evaluation resolution.** Three methods halve the native video internally, in three different places; a fourth had no N3DV path at all. | Frames are extracted once at 1352×1014 and every method is told not to halve again. `eval_resolution` is recorded per entry (§2.1). |
| 2 | **Test split.** Each repository selects the held-out camera differently (index 0 of the pose file, the first camera in alphabetical order, the first `duration` entries after sorting). | All three rules select `cam00`, the dataset's own held-out camera; the fourth method is given the same split explicitly. The result is 50 views everywhere, with `MAX_EVAL_VIEWS = 0` (§2.2). |
| 3 | **Temporal window.** 300 frames for two methods, 50-frame chunks for one, per-frame cost for the fourth. | One window, 50 frames, for all four; recorded in the JSON and in the run name (§2.3). |
| 4 | **Initial point cloud.** 4DGaussians uses the dynerf release's `points3D_downsample2.ply`, fudan-zvg a COLMAP dense fusion, Dynamic 3D Gaussians a cloud it never had for this dataset — three different clouds, whose effect would have been attributed to the representations. | Notebooks `04`, `05` and `06` run **one shared COLMAP stage** on the first frame, with the same settings and the poses taken from `poses_bounds.npy` rather than estimated. Each method then applies **its own** post-processing to that one cloud: voxel downsampling to ≤ 40 000 points for Wu et al. (the algorithm of their `scripts/downsample_point.py`, reimplemented in numpy because it calls `open3d`), random subsampling to `num_pts` for fudan-zvg, none for Luiten et al. Note that every one of those is a **cap, not a target**: nothing upsamples a cloud that is already smaller, so the size of this one cloud is the initialisation of all three. It is recorded as `init_num_points` next to `init_point_cloud`. Spacetime Gaussians is the exception, see 4.2 #5. |
| 5 | **LPIPS backend.** The other three repositories bundle `lpipsPyTorch`; Dynamic 3D Gaussians does not, and would have fallen back to the pip `lpips` package, putting a systematic offset on one column of the table. | All four multi-view notebooks force `LPIPS_BACKEND = "lpips-pip"` with the `[-1, 1]` rescaling that package expects, and record the resolved backend in the summary. This differs from the monocular study, which used the bundled package; the two studies are reported separately. |
| 6 | **Background convention.** In the monocular study the black background was forced on all three methods by fudan-zvg's alpha premultiplication. | Moot here: N3DV frames are real captures with no alpha channel, so no ground truth is premultiplied and every method trains on `white_background = False` with an unused background tensor. |
| 7 | **Metric definitions.** Four repositories, four copies of `psnr`, `ssim` and `l1`. | Checked to be the same formulas: Dynamic 3D Gaussians' `external.calc_psnr` is `20·log10(1/√MSE)`, its `calc_ssim` is the same windowed SSIM as `utils.loss_utils.ssim`, and `helpers.l1_loss_v1` is `mean|x − y|`. Each method is still measured with its own modules, as in the monocular study. |
| 8 | **Evaluation schedule.** | `EVAL_EVERY_N_ITERS = 1000` in three notebooks; for the frame-by-frame method the grid is set to its per-frame budget so that one sample lands at the end of every frame (§5.2). |

### 4.2 Declared, not removed

| # | Difference | Why forcing it would distort the comparison | Where it is visible |
|---|---|---|---|
| 1 | **Batch size / images seen.** The official per-scene value everywhere: 4 views per step for fudan-zvg (gradient accumulation); for 4DGaussians 4 on `coffee_martini` and `flame_salmon_1` and 2 on the other four scenes, as its `arguments/dynerf/<scene>.py` files set; 2 for Spacetime Gaussians; 1 for Dynamic 3D Gaussians. | Identical to the monocular case ([METHODOLOGY.md](METHODOLOGY.md) §3.5): a step is not the same unit of work, and the per-scene batch is part of each method's tuning. | The `images_seen` column (steps × batch), which the analysis uses as the budget axis. |
| 2 | **Optimisation structure.** Three methods optimise the whole window at once; Dynamic 3D Gaussians optimises frame by frame, with a fixed budget per frame. | Forcing 30 000 total steps would give it ~408 steps per frame against the 2 000 its authors chose, i.e. a number about a method nobody runs. Its budget is a knob (`D3DG_BUDGET_MODE`), defaulting to the native schedule. | `budget_mode`, `iters_first_timestep`, `iters_per_timestep` in the JSON; `timesteps_done` per entry. |
| 3 | **Model variant.** Spacetime Gaussians ships `ours_lite` (direct RGB) and `ours_full` (an MLP colour decoder). | `ours_lite` is the current default, because no other method in this comparison has a neural colour decoder. `ours_full` is fully supported: `oursfull.py` saves the decoder to `point_cloud.pt` and the storage report counts it. Which one to benchmark is an open decision ([OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) §3). `STG_MODEL` switches. | `stg_model` in the JSON. |
| 4 | **Environment map.** fudan-zvg's official configs enable a 500×500 learned environment map on `coffee_martini` and `flame_salmon_1` and disable it on the other four scenes. | It is part of the official per-scene configuration of that method for those scenes. | `env_map_res` in the JSON; the environment map is **counted in the model storage**, unlike the monocular study where it was always empty. |
| 5 | **Per-frame initialisation.** Spacetime Gaussians needs a COLMAP model per frame, not one for the window. | The per-frame clouds are what gives its primitives a temporal spread; replacing them with a single-frame cloud would remove a component of the method. | `init_point_cloud: "sparse_per_frame"` in the JSON, against `"dense"` or `"sparse"` for the others. |
| 6 | **Per-camera colour correction.** Dynamic 3D Gaussians learns an exposure gain and offset (`cam_m`, `cam_c`) for each of the 20 training cameras. | It is part of the method, and the other three have nothing like it. The **held-out camera has no correction**, so its test renders are scored raw — which is the honest thing to do and is what the method can produce for an unseen camera. | Declared here; the correction is in `params.npz`. |
| 7 | **Storage definitions.** A `.ply` + a `.pth` for 4DGaussians, model tensors of a checkpoint for fudan-zvg, a `.ply` for Spacetime Gaussians, a single `params.npz` for Dynamic 3D Gaussians whose size **grows with the window**. | Each is what that method actually needs on disk to render; a common definition does not exist. | `storage.required_files` in each JSON; for the frame-by-frame method, `storage.arrays` gives the shape of every array. |
| 8 | **Densification hyper-parameters** and therefore the number of Gaussians. | Unchanged from the monocular study: the count measures a hyper-parameter choice, not a property of the representation. | Declared, as in [METHODOLOGY.md](METHODOLOGY.md) §9. |

## 5. What had to be added, per method

Every addition beyond configuration is listed here. Three of the four notebooks change no upstream code at all beyond the monitor hook.

### 5.1 4DGaussians (notebook 05): one reader constant

`scene/neural_3D_dataset_NDC.py` hard-codes `countss = 300` and `scene/dataset_readers.py` hard-codes `maxtime=300`. Both are rewritten to read `BENCH_N3DV_FRAMES`, defaulting to 300 when it is unset.

This is **the only change to reader code in the four notebooks**, and it is necessary rather than convenient: the reader sets each frame's timestamp to `idx / countss`. With 50 frames and `countss` left at 300 the window would occupy `t ∈ [0, 0.163]`, i.e. one sixth of the temporal axis of the HexPlane grid, while the other three methods span their full temporal range — the runs would not be comparable. The other three get their window from a configuration value (`time_duration`, `--duration`, the converter), so no patch is needed there.

Data preparation: the extracted frames are linked into the `<cam>/images/%04d.png` layout the reader expects, with an empty placeholder `.mp4` next to each one (the reader globs `cam*.mp4` only to enumerate the cameras and check their number against `poses_bounds.npy`; it opens a video only when the image folder is missing).

### 5.2 Dynamic 3D Gaussians (notebook 04): the largest adaptation

The authors run on Panoptic Sports, a dataset they prepared and released ready to use; **their data-preparation code and their novel-view evaluation code are not released**. Four things had to be built.

1. **Panoptic-format conversion.** `train_meta.json`, `test_meta.json`, `ims/<cam>/<t>.png`, `seg/<cam>/<t>.png` and `init_pt_cld.npz`, from `poses_bounds.npy` and the extracted frames. The poses are converted to the OpenCV convention the repository documents.
2. **World frame.** `get_loss` contains a floor loss, `clamp(y, min=0)`, which the README justifies with "we know the ground-plane of the scenes we are using": Panoptic is y-down with the floor at y = 0, so the term penalises geometry *below* the floor and is inactive for a well-placed scene. `get_dataset` also hard-codes `near = 1.0`. The converter therefore rotates the world so that +y is down (from the mean camera up-vector), translates it so the lowest point of the initial cloud sits at y = 0, and scales it so the LLFF near bound maps to 1. This is a change of orientation and units only; no relative geometry is altered.
3. **Segmentation masks.** The method needs a binary dynamic/static mask per image — its `seg` loss supervises them, and `is_fg` decides which primitives the rigidity, rotation and isometry losses act on and which are anchored as background. **N3DV ships none.** They are estimated as the pixels that differ from each camera's temporal median by more than `SEG_DIFF_THRESHOLD`, cleaned with a morphological open/close and a minimum-component-area filter. A point of the initial cloud is labelled foreground when it projects inside the per-camera union-over-time mask in more than half of the cameras that see it. **This is the largest addition in the four notebooks and it bounds what the method can do here**: the mask quality is a property of this preparation, not of the method.
4. **Two code changes.** `helpers.o3d_knn` is reimplemented on `scipy.spatial.cKDTree`, because Open3D has no wheel for the Python version Colab ships (same inputs, same two outputs, pristine copy kept). The per-frame iteration count `10000 if is_initial_timestep else 2000` is made to read two environment variables defaulting to those values. A separate `train_one.py` runs one sequence, because `train.py`'s `__main__` hard-codes the six Panoptic sequences; `train.py` itself is otherwise untouched.

**How the test metric is defined for a frame-by-frame method.** At every sampling point the monitor asks the model for **the same 50 test views the other three methods are asked for**. A frame already finished is rendered from its snapshot in `output_params`; the current and the still unreached frames are rendered from the live parameters, which is all the model can produce for them. The curve therefore starts poor and improves as the sequence is covered, and at the end of the run the value is exactly the mean over the 50 test views, directly comparable with the other three. Losses, weights, densification, learning rates, the constant-velocity per-frame initialisation and the two-phase schedule are upstream.

### 5.3 4DGS native-4D (notebook 06): a reimplemented converter

`scripts/n3v2blender.py` extracts all 300 frames of all 21 cameras at full 2704×2028 and then runs COLMAP including dense MVS — tens of gigabytes and hours of CPU. The notebook writes the same `transforms_train.json` / `transforms_test.json` / `images/` / `points3d.ply` from the frames of cell 3 and the cloud of cell 3.4, with **the pose arithmetic reimplemented line by line from that script**: the same LLFF→NeRF permutation, the same `up`-vector alignment, the same recentring on the point closest to all camera axes, the same rescaling to an average radius of 4. The same world transform is applied to the point cloud, so cameras and geometry stay in one frame.

Two deliberate config deviations, both recorded: `resolution: 1` instead of `2` (the frames are already halved, see §2.1) and `time_duration: [0, (NUM_FRAMES−1)/30]` instead of `[0, 10]` (ten seconds is the 300-frame sequence). Everything else is the official per-scene YAML. `exhaust_test: False`, as in the monocular study.

### 5.4 Spacetime Gaussians (notebook 07): the upstream stage, with our frames

`script/pre_n3d.py` is called function by function — `preparecolmapdynerf`, `convertdynerftocolmapdb`, `getcolmapsinglen3d` are the repository's own, imported rather than copied — with one substitution: the frames come from cell 3 instead of being decoded again at full resolution by `extractframes`, and the intrinsics are divided by the same factor. No repository code is modified beyond the monitor hook. This is the slowest preparation of the four (`NUM_FRAMES` COLMAP runs on 21 images each); it is cached per scene and resumable.

## 6. Iteration budget

| Method | static warm-up | main optimisation | total | note |
|---|---|---|---|---|
| 4DGaussians (Wu et al.) | 3 000 (coarse) | 27 000 (fine) | 30 000 | official N3DV default is 3 000 + 14 000 |
| 4DGS native-4D (fudan-zvg) | none | 30 000 | 30 000 | official N3DV default is 30 000 |
| Spacetime Gaussians (Li et al.) | none | 30 000 | 30 000 | official default trains 30 000 and the N3DV configs evaluate the 25 000 snapshot (`test_iteration: 25000`) |
| Dynamic 3D Gaussians (Luiten et al.) | 10 000 on the first frame | 2 000 per later frame | 108 000 at 50 frames | see §4.2 #2 |

As in the monocular study the schedules are **not** rescaled: learning rates and densification stay at the upstream values. 4DGaussians is run past its N3DV default and this is declared as a limitation.

## 7. Protocol B

Unchanged in mechanism: stop when the evaluation L1 reaches a per-scene target common to the methods, sampled every 30 s, two consecutive hits, cost read at the **first crossing**. The calibration rule is the one of [PROTOCOL_B_CALIBRATION.md](PROTOCOL_B_CALIBRATION.md): `target(scene) = 1.05 × max over methods (minimum eval L1 on that method's Protocol A curve)`.

**The targets ship as `None`.** They are derived from Protocol A results, and no multi-view Protocol A run exists yet; with `None` the Protocol B loop skips the scene rather than training against an invented bar. Cell 4.3 of each notebook prints that method's candidates once its Protocol A loop has run.

**One caveat specific to the frame-by-frame method.** Its test curve improves as the window is covered, so a Protocol B stop truncates the sequence at the frame where the target was met, and the model then represents fewer than 50 frames. `timesteps_done` in the last entry makes that visible, and a Protocol B run of notebook `04` must be read with it in hand. The comparable Protocol B figures are the three whole-window methods; for the fourth, Protocol A is the primary comparison.

## 8. Declared limitations

Everything in [METHODOLOGY.md](METHODOLOGY.md) §9 that is not dataset-specific still applies. In addition:

* **No results yet.** The four notebooks, the protocol and the analysis path exist; no multi-view training run has been made, so `docs/RESULTS.md` has no multi-view section and `results/n3dv/` is empty.
* **The 50-frame window breaks comparability with the literature** (§2.3). No published N3DV number applies to these runs.
* **The segmentation masks of notebook `04` are an addition of this project**, not of its authors, and their quality bounds that method's results (§5.2).
* **Dynamic 3D Gaussians is being used outside the dataset family its authors validated it on**, with a world frame and a floor plane inferred rather than known. Its numbers say what the method does on N3DV under this preparation, not what its authors would report.
* **One reader constant is patched** in notebook `05` (§5.1) — a class of change the monocular study did not need.
* **The initial point cloud may be sparse rather than dense, and smaller than upstream's**: the shared COLMAP stage asks for `patch_match_stereo` + `stereo_fusion`, which needs a CUDA-enabled COLMAP; where the Colab image ships one built without CUDA the stage falls back to the triangulated sparse points and records `init_point_cloud: "sparse"`. Because each method's `num_pts` is a cap and not a target, a sparse fallback means fudan-zvg starts from far fewer than its nominal 300 000 primitives and 4DGaussians from fewer than 40 000, which is a handicap relative to what their authors run. Both the flag and the point count (`init_num_points`) are in every benchmark JSON; runs must not be compared across them.
* **One run per configuration**, without repeated seeds, as before.
* The per-camera exposure model of notebook `04` is not applied to the held-out camera (§4.2 #6), which is correct but means its test renders carry no exposure compensation while its training renders do.
