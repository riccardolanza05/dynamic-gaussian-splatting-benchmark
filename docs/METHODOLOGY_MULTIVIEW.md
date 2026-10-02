# Methodology, multi-view study (Neural 3D Video)

This document is the multi-view counterpart of [METHODOLOGY.md](METHODOLOGY.md). It describes how the benchmark was extended from the monocular D-NeRF setting to the multi-camera setting of *Neural 3D Video Synthesis from Multi-View Video* (N3DV), which methods it covers, what had to be aligned across the repositories, and **which differences between the methods could be removed and which could only be declared**.

The instrumentation, the metric definitions and Protocol B are those of the monocular study: [METHODOLOGY.md](METHODOLOGY.md) §2 (instrumentation) and §4 (definition of the evaluation loss) apply here verbatim and are not repeated. **Protocol A differs**: each method runs at its own official budget instead of a common one (§6).

Status: the notebooks and the protocol are in place; **no multi-view training run has been made yet**, so there are no results and the Protocol B targets are empty by construction. See §8.

Two companion pages: [OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) records the choices behind this protocol — those taken, with their reasons, and those still open (the GPU, Protocol B on the full sequence, a temporally subsampled window, the LPIPS backend); [DISCLOSURES_MULTIVIEW.md](DISCLOSURES_MULTIVIEW.md) collects what a reader of the results has to be careful about, and needs no decision.

## 1. Scope

**Goal.** The same controlled comparison, on the setting the monocular study could not reach: synchronised multi-camera video, where every instant is observed by many cameras at once.

**Methods benchmarked** (three of the five studied):

| Method | Repository | Representation | Notebook | Status before this extension |
|---|---|---|---|---|
| 4DGaussians / HexPlane (Wu et al., CVPR 2024) | [`hustvl/4DGaussians`](https://github.com/hustvl/4DGaussians) | canonical space + factorised HexPlane grid + light decoder | [`05`](../notebooks/05_4dgaussians_wu_n3dv.ipynb) | benchmarked on monocular D-NeRF |
| 4DGS, native 4D primitives (Yang et al., ICLR 2024) | [`fudan-zvg/4d-gaussian-splatting`](https://github.com/fudan-zvg/4d-gaussian-splatting) | native 4D Gaussians with finite temporal extent | [`06`](../notebooks/06_4dgs_native4d_fudan_n3dv.ipynb) | benchmarked on monocular D-NeRF |
| Spacetime Gaussians (Li et al., CVPR 2024), in both released variants | [`oppo-us-research/SpacetimeGaussians`](https://github.com/oppo-us-research/SpacetimeGaussians) | temporal radial-basis opacity + polynomial motion, per-frame COLMAP init | [`07`](../notebooks/07_spacetime_gaussians_li_n3dv.ipynb) | studied, never benchmarked |

Deformable-3DGS is left out: it targets single-camera scenes (NeRF-DS, HyperNeRF, D-NeRF) and has no multi-view data path.

### 1.1 Dynamic 3D Gaussians: studied, then set aside

Dynamic 3D Gaussians (Luiten et al., 3DV 2024) accepts multi-camera input and was part of this study until 2026-10-02, with its own notebook (`04`). It was set aside before any run was made. The reasons, with the numbers, are in [OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md#dynamic-3d-gaussians-set-aside-2026-10-02); in short:

* **The only published N3DV result of the method** is in the Spacetime Gaussians paper (Appendix B, Table 6), made by those authors with their own tuning: 30.67 dB PSNR, 0.099 LPIPS, 2772 MB of model and 460 FPS, against 31.59–32.05 dB, 0.044–0.047 LPIPS, 103–200 MB and 140–310 FPS for Spacetime Gaussians in the same table. It is last in quality, its LPIPS is about twice as bad, and its model is 14 to 27 times larger; it wins only on rendering speed.
* **That advantage does not matter for the purpose of this project**, a fluid playback: above roughly 120 FPS the rendering is already fluid, and the difference between 140 and 460 FPS is not perceptible on an ordinary display. Rendering speed is also not one of the quantities this benchmark measures.
* **Including it would have shaped the whole study around it.** It optimises frame by frame, at a fixed cost per frame (108 000 steps at 50 frames, 608 000 at 300), which is what had forced a 50-frame main window. Its authors never ran it on N3DV and did not release their data-preparation code; it needs foreground masks and a floor plane that N3DV does not provide; and stopping it early truncates the sequence, so Protocol B would not have been comparable.
* **It is a tracking method.** Its paper is titled *Tracking by Persistent Dynamic View Synthesis*: it accepts constraints that cost image quality in exchange for dense 3D tracking, which this benchmark does not measure.

The method remains one of the five studied ([METHODOLOGY.md](METHODOLOGY.md) §1). Nothing about it was measured here.

### 1.2 Dataset and hardware

**Dataset.** N3DV, six scenes (`coffee_martini`, `cook_spinach`, `cut_roasted_beef`, `flame_salmon_1`, `flame_steak`, `sear_steak`), about 20 synchronised cameras (the release filtered out unsynchronised streams, so the number varies by scene), 300 frames at 30 fps, native 2704×2028.

**Hardware.** A single cloud GPU, on a [lightning.ai](https://lightning.ai) Studio. The monocular study ran on a Google Colab Tesla T4; the multi-view runs are not tied to that machine, and the notebooks work unchanged on a Studio, on Colab or on any Linux machine with an NVIDIA card (see the [running notes in the README](../README.md#running-on-a-cloud-gpu-over-ssh)).

**Which GPU is itself a decision**, and it has to be taken before the first run: training time and peak VRAM are two of the monitored metrics, and both are properties of the pair (method, GPU). Every run of the study must use the same GPU type, and that type must be reported next to the numbers. See [OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) §4.

## 2. The three conventions that make the comparison possible

### 2.1 Resolution: 1352×1014

Half of the native 2704×2028. This is not a compromise: it is the resolution the **official N3DV configuration of every one of the three methods** uses. 4DGaussians' `Neural3D_NDC_Dataset` hard-codes `img_wh = (1352, 1014)`; fudan-zvg's `configs/dynerf/*.yaml` and Spacetime Gaussians' `configs/n3d_*/*.json` both set `resolution: 2` on the native video.

The frames are therefore extracted **once**, at 1352×1014, by a cell shared by the three notebooks, and each method is then told not to halve them again (`resolution: 1` for fudan-zvg, `--resolution 1` for Spacetime Gaussians; 4DGaussians resizes to the size the frames already have). The effective resolution is the official one; only the place where the downscaling happens moves. Every benchmark entry records `eval_resolution`, so the guarantee is checked on every run.

### 2.2 Test split: `cam00`, the whole window

The N3DV README states that `cam00` is the centre reference camera, held out for testing. All three methods follow it, each by its own rule (index 0 of the pose file, camera 0 of the conversion script, the first camera in sorted order), and all three rules select `cam00`. The test set is one view per frame of the window, so `MAX_EVAL_VIEWS` stays at `0` (evaluate everything) and the evaluation L1 is directly comparable across methods.

### 2.3 Temporal window: 300 frames, and a 50-frame secondary study

**The main study uses the whole sequence, 300 frames (10 s at 30 fps), with 300 test views.** This is what the three papers report, so the results can be set next to the published ones: the check against the literature that [METHODOLOGY.md](METHODOLOGY.md) §7 performs for the monocular study has a counterpart here.

* **4DGaussians and 4DGS native-4D** build one model for the whole window.
* **Spacetime Gaussians covers a sequence in blocks of 50 frames** by construction (`duration: 50` in every official N3DV config). Its 300-frame run is six independent models, as in its paper, merged into one result (§5.3).

**A secondary study uses the first 50 frames** (1.67 s, 50 test views), with the same three methods. It is one Spacetime Gaussians block, so every method is a single model there, which is what Protocol B needs (§7). No published number is comparable with it.

`NUM_FRAMES` is a configuration value in all three notebooks, it is written into **every** benchmark JSON (`num_frames`) and into every run folder name (`<scene>_f300_iters30000`), and `run_is_complete()` refuses to reuse a run made with a different window. The analysis reads the two windows as two separate studies. Runs from different windows therefore cannot be mixed by accident.

Until 2026-10-02 the main window was 50 frames, because Dynamic 3D Gaussians could not be run on more (§1.1).

## 3. Instrumentation: the same monitor, one new hook point

The monitor is the one described in [METHODOLOGY.md](METHODOLOGY.md) §2: a few lines appended to `train.py`, before `if __name__ == "__main__":`, that wrap a function the training loop already calls at every iteration. Upstream behaviour, hyper-parameters and command line are untouched, a pristine copy is kept as `train.py.orig`, and without `BENCH_CONFIG` the file behaves exactly as upstream.

Two of the three repositories expose the same `training_report(...)` as the monocular ones, and the monitor binds its arguments with `inspect.signature().bind()` as before. **The third has no such function.**

| Repository | Wrapped function | Where the loop state comes from |
|---|---|---|
| hustvl/4DGaussians | `training_report(...)` | the signature |
| fudan-zvg/4d-gaussian-splatting | `training_report(...)` | the signature |
| oppo-us-research/SpacetimeGaussians | `controlgaussians(...)` (from `helper_train`) | signature + calling frame |

`controlgaussians` is a module-level name that the loop resolves at call time, so rebinding it in `train.py` works exactly as rebinding `training_report` does, and it is called once per iteration inside the loop's `torch.no_grad()` block. What its narrower signature does not carry — the render function, the pipeline parameters, the background, the current loss — is read from the **calling frame** with `sys._getframe`. The frame is only read, never written.

**The monitor core is byte-identical in the three notebooks.** It is delimited by `# === BENCHMARK MONITOR CORE ===` / `# === END BENCHMARK MONITOR CORE ===`, and everything repository-specific is in a glue block appended after it, behind ten named functions: `_stage_is_evaluable`, `_stage_of`, `_global_iteration`, `_num_gaussians`, `_train_batch_loss`, `_force_eval`, `_iter_eval_views`, `_render_pair`, `_save_model`, `_maybe_track_best`. A check asserts the identity of the core across the three files, so "the same methodology" is checked and not only asserted.

Fields added to the JSON of the monocular study: `num_frames`, `held_out_camera` and `dataset` in every run; `preparation_time_s` and `scene_data_mb` (the time spent preparing the scene in that invocation, and the disk its prepared data takes), which are not metrics of the method but are needed to budget the runs; and, in the run's config, what upstream itself would have done on that scene (`official_iterations`, `official_coarse_iterations`, `official_test_iteration`). Nothing was removed.

## 4. Differences between the methods: removed, or declared

A difference is **removed** when it can be equalised without changing what a method is, and **declared** when equalising it would report a number about a method nobody runs.

### 4.1 Removed

| # | Difference | How it was removed |
|---|---|---|
| 1 | **Evaluation resolution.** The three methods halve the native video internally, in three different places. | Frames are extracted once at 1352×1014 and every method is told not to halve again. `eval_resolution` is recorded per entry (§2.1). |
| 2 | **Test split.** Each repository selects the held-out camera by a different rule. | All three rules select `cam00`, the dataset's own held-out camera, over the whole window, with `MAX_EVAL_VIEWS = 0` (§2.2). |
| 3 | **Temporal window.** | One window per study, the same for the three methods: 300 frames in the main study, 50 in the secondary one; recorded in the JSON and in the run name (§2.3). At 300 frames this is each method's official window. |
| 4 | **Initial point cloud.** 4DGaussians uses a dense COLMAP fusion downsampled by its own script, fudan-zvg a dense COLMAP fusion built by its own script — two reconstructions, whose difference would have been attributed to the representations. | Notebooks `05` and `06` run **one shared COLMAP stage** on the first frame, with the same settings and the poses taken from `poses_bounds.npy` rather than estimated. Each method then applies **its own** post-processing to that one cloud: voxel downsampling to ≤ 40 000 points for Wu et al. (the algorithm of their `scripts/downsample_point.py`, reimplemented in numpy because it calls `open3d`), random subsampling to `num_pts` for fudan-zvg. Both are a **cap, not a target**: nothing upsamples a cloud that is already smaller. The size is recorded as `init_num_points` next to `init_point_cloud`. Spacetime Gaussians is the exception, see 4.2 #6. |
| 5 | **LPIPS backend.** | All three notebooks use the pip `lpips` package with the `[-1, 1]` rescaling it expects, and record the resolved backend in the summary. This differs from the monocular study, which used each repository's bundled package; the two studies are reported separately. Whether to keep it is an open, low-stakes choice ([OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) §9). |
| 6 | **Background convention.** In the monocular study the black background was forced on all methods by fudan-zvg's alpha premultiplication. | Moot here: N3DV frames are real captures with no alpha channel, so no ground truth is premultiplied and every method trains on `white_background = False`. |
| 7 | **Metric definitions.** Three repositories, three copies of `psnr`, `ssim` and `l1`. | Each method is measured with its own modules, as in the monocular study: `utils.image_utils.psnr`, `utils.loss_utils.ssim` and `utils.loss_utils.l1_loss`, which all three repositories carry over from the 3D Gaussian Splatting code base. That the three copies are still identical is to be confirmed on the first run, by checking one rendered pair against a single reference implementation. |
| 8 | **Evaluation schedule.** | Every run is sampled about 30 times: every 500 fine steps for 4DGaussians, every 1 000 steps for the other two (§6). |

### 4.2 Declared, not removed

| # | Difference | Why forcing it would distort the comparison | Where it is visible |
|---|---|---|---|
| 1 | **Iteration budget.** 3 000 + 14 000 steps for 4DGaussians, 30 000 for 4DGS native-4D, 30 000 for Spacetime Gaussians. | Each is the budget its authors use on this dataset, and the number its paper reports comes from it. A common budget would run 4DGaussians at nearly twice its own (§6). | `max_iterations`, `iteration_offset`, `official_iterations` in the JSON; `total_iterations` per entry. |
| 2 | **Batch size / images seen.** The official per-scene value everywhere: 4 views per step for fudan-zvg; for 4DGaussians 4 on `coffee_martini` and `flame_salmon_1` and 2 on the other four scenes, as its `arguments/dynerf/<scene>.py` files set; 2 for Spacetime Gaussians. | Identical to the monocular case ([METHODOLOGY.md](METHODOLOGY.md) §3.5): a step is not the same unit of work, and the per-scene batch is part of each method's tuning. Together with #1, a Protocol A run sees 34 000 or 68 000 images (4DGaussians), 120 000 (4DGS native-4D) or 60 000 per model (Spacetime Gaussians). | The `images_seen` column (steps × batch), which the analysis uses as a budget axis. |
| 3 | **Model variant.** Spacetime Gaussians ships `ours_lite` (direct RGB) and `ours_full` (an MLP colour decoder). | **Both are benchmarked, as two rows.** `ours_full` is the method as its paper presents it; `ours_lite` is the variant comparable with the other two, neither of which has a neural colour decoder. If both prove too expensive, only `ours_lite` is run ([OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) §3). | `stg_model` in the JSON; a `_full` tag in the run folder. |
| 4 | **Models per scene.** Spacetime Gaussians is six 50-frame models at 300 frames; the other two are one model. | It is how the method covers a sequence, and how its paper's numbers are made. | `blocks` in the merged JSON; time, steps, images and storage are sums over the six (§5.3). |
| 5 | **Environment map.** fudan-zvg's official configs enable a 500×500 learned environment map on `coffee_martini` and `flame_salmon_1` and disable it on the other four scenes. | It is part of the official per-scene configuration of that method. | `env_map_res` in the JSON; the environment map is **counted in the model storage**, unlike the monocular study where it was always empty. |
| 6 | **Per-frame initialisation.** Spacetime Gaussians needs a COLMAP model per frame, not one for the window. | The per-frame clouds are what gives its primitives a temporal spread; replacing them with a single-frame cloud would remove a component of the method. | `init_point_cloud: "sparse_per_frame"` in the JSON, against `"dense"` or `"sparse"` for the others. |
| 7 | **Storage definitions.** A `.ply` + a `.pth` for 4DGaussians, the model tensors of a checkpoint for fudan-zvg, a `.ply` (plus a `.pt` decoder for `ours_full`) for Spacetime Gaussians. | Each is what that method actually needs on disk to render; a common definition does not exist. | `storage.required_files` in each JSON. |
| 8 | **Densification hyper-parameters** and therefore the number of Gaussians. | Unchanged from the monocular study: the count measures a hyper-parameter choice, not a property of the representation. | Declared, as in [METHODOLOGY.md](METHODOLOGY.md) §9. |

## 5. What had to be added, per method

Every addition beyond configuration is listed here. Two of the three notebooks change no upstream code at all beyond the monitor hook.

### 5.1 4DGaussians (notebook 05): one reader constant

`scene/neural_3D_dataset_NDC.py` hard-codes `countss = 300` and `scene/dataset_readers.py` hard-codes `maxtime=300`. Both are rewritten to read `BENCH_N3DV_FRAMES`, defaulting to 300 when it is unset.

**In the main study the patched values are the upstream ones** (300 frames), so the patch changes nothing there. It matters for the 50-frame study: the reader sets each frame's timestamp to `idx / countss`, so with `countss` left at 300 a 50-frame window would occupy `t ∈ [0, 0.163]`, one sixth of the temporal axis of the HexPlane grid, while the other two methods span their full temporal range. This is the only change to reader code in the three notebooks.

Data preparation: the extracted frames are linked into the `<cam>/images/%04d.png` layout the reader expects, with an empty placeholder `.mp4` next to each one (the reader globs `cam*.mp4` only to enumerate the cameras and check their number against `poses_bounds.npy`; it opens a video only when the image folder is missing).

**The repository is followed where it disagrees with the paper.** The paper's appendix (A.1) says the batch size is 1 and the dense point cloud is downsampled below 100 000 points; the repository's N3DV configs use batch 4 or 2 and its README says the cloud is downsampled to fewer than 40 000 points. The runs use the repository's values.

### 5.2 4DGS native-4D (notebook 06): a reimplemented converter

`scripts/n3v2blender.py` extracts all 300 frames of all cameras at full 2704×2028 and then runs COLMAP including dense MVS. The notebook writes the same `transforms_train.json` / `transforms_test.json` / `images/` / `points3d.ply` from the frames of cell 3 and the cloud of cell 3.4, with **the pose arithmetic reimplemented line by line from that script**: the same LLFF→NeRF permutation, the same `up`-vector alignment, the same recentring on the point closest to all camera axes, the same rescaling to an average radius of 4. The same world transform is applied to the point cloud, so cameras and geometry stay in one frame.

Config values written by the notebook, both recorded: `resolution: 1` instead of `2` (the frames are already halved, see §2.1), and `time_duration: [0, NUM_FRAMES / 30]`. At 300 frames the second is exactly the official `[0, 10]`; at 50 frames it is `[0, 1.667]`. Frame times are `frame_index / 30`, as in `n3v2blender.py`. Everything else is the official per-scene YAML. `exhaust_test: False`, as in the monocular study.

### 5.3 Spacetime Gaussians (notebook 07): the upstream stage, with our frames

`script/pre_n3d.py` is called function by function — `preparecolmapdynerf`, `convertdynerftocolmapdb`, `getcolmapsinglen3d` are the repository's own, imported rather than copied — with one substitution: the frames come from cell 3 instead of being decoded again at full resolution by `extractframes`, and the intrinsics are divided by the same factor. No repository code is modified beyond the monitor hook. This is the slowest preparation of the three (`NUM_FRAMES` COLMAP runs on about 20 images each); it is cached per scene and resumable.

**Beyond 50 frames: one model per block, as the paper does.** In the main study the scene is trained as six independent 50-frame models. Block *k* is an ordinary upstream run on `colmap_<50k>` with `--duration 50`: the reader walks `colmap_<50k>` … `colmap_<50k+49>`, holds out `cam00` for those frames and rescales time to the block. Its monitor JSON is kept under `<run>/blocks/block_<k>/`. The merged result follows the paper's definitions where it has one: quality (PSNR, SSIM, LPIPS, L1) is the mean over the 300 test views; training time, steps, images seen, Gaussians and storage are summed over the six models; peak VRAM is their maximum. `iteration` stays the per-block step, and `total_iterations` is the sum. Protocol B is not run in blocks (§7).

**Two variants, two runs.** The notebook trains one variant per execution (`STG_MODEL`, `ours_lite` by default). `ours_full` runs go to their own folders. The per-frame COLMAP models are reused by the second variant only if the first run kept the scene data (`DELETE_FRAMES_AFTER_TRAIN=false`).

## 6. Protocol A: each method at its official budget

| Method | static warm-up | main optimisation | total | sampling | source |
|---|---|---|---|---|---|
| 4DGaussians (Wu et al.) | 3 000 (coarse) | 14 000 (fine) | 17 000 | every 500 fine steps: 28 samples + baseline | `arguments/dynerf/default.py`, inherited by every scene |
| 4DGS native-4D (fudan-zvg) | none | 30 000 | 30 000 | every 1 000 steps: 30 samples + baseline | `configs/dynerf/<scene>.yaml`, every scene |
| Spacetime Gaussians (Li et al.) | none | 30 000 per model | 30 000 per model, 180 000 for the six models of a 300-frame scene | every 1 000 steps: 30 samples + baseline, per model | default of `thirdparty/gaussian_splatting/arguments/__init__.py` |

**Why not a common budget.** The monocular study gave every method 30 000 steps and found that a step is not a common unit of work ([METHODOLOGY.md](METHODOLOGY.md) §3.5): the batch sizes differ, so equal steps are unequal training. On N3DV a common 30 000 would in addition run 4DGaussians at nearly twice its official budget. Protocol A therefore asks a different question here than in the monocular study: not *"at equal steps, which method reconstructs best?"* but *"run as its authors run it, how well does each method reconstruct, and at what cost?"*. The methods are compared on images seen and on training time, not on steps.

**Schedules are untouched.** Learning rates and densification stay at the upstream values, and each run records the official budget of its scene as read back from the official config (`official_iterations`). A run made with a different `MAX_ITERATIONS` says so in its log.

**Spacetime Gaussians is read twice.** Its official flow trains 30 000 steps and evaluates one snapshot per scene, set by `test_iteration` in the config: 25 000 on `cook_spinach`, `cut_roasted_beef`, `flame_steak` and `sear_steak`, 10 000 on `coffee_martini`, 12 000 on `flame_salmon_1`. Both the end of the run and that snapshot are reported; the snapshot is the reading to set next to the paper (whose own long-sequence table lists 12K iterations per model for `flame_salmon`). All three values are on the sampling grid, and each run records its own (`official_test_iteration`).

**Sampling.** About 30 samples per run, so that curves of different length have a comparable resolution. Each sample evaluates the whole test split; that time is excluded from the training time, as in the monocular study.

## 7. Protocol B

Unchanged in mechanism: stop when the evaluation L1 reaches a per-scene target common to the methods, sampled every 30 s, two consecutive hits, cost read at the **first crossing**. The calibration rule is the one of [PROTOCOL_B_CALIBRATION.md](PROTOCOL_B_CALIBRATION.md): `target(scene) = 1.05 × max over methods (minimum eval L1 on that method's Protocol A curve)`. The rule never assumed equal budgets, so it applies unchanged to curves of different length; the bar is the worst of the methods' bests, each run as its authors run it.

**Protocol B is run on the 50-frame window.** There every method is a single model. At 300 frames Spacetime Gaussians is six independent models, for which "the first time the target is reached" is not defined; notebook `07` refuses a Protocol B run in blocks. How Protocol B could be carried to the main window is an open decision ([OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) §7). A set of targets belongs to one window: targets calibrated on 50-frame runs are not valid for 300 frames.

**The targets ship as `None`.** They are derived from Protocol A results, and no multi-view Protocol A run exists yet; with `None` the Protocol B loop skips the scene and does not train against an invented bar. Cell 4.3 of each notebook prints that method's candidates once its Protocol A loop has run.

## 8. Declared limitations

Everything in [METHODOLOGY.md](METHODOLOGY.md) §9 that is not dataset-specific still applies. In addition:

* **No results yet.** The three notebooks, the protocol and the analysis path exist; no multi-view training run has been made, so `docs/RESULTS.md` has no multi-view section and `results/n3dv/` is empty. Nothing that needs CUDA, COLMAP or the dataset has been executed.
* **Protocol A is not an equal-budget comparison.** Steps and images seen differ between the methods by design (§6). A method may be ahead because its authors chose a larger budget.
* **Dynamic 3D Gaussians was set aside without being run** (§1.1). The study says nothing about it.
* **The main study has no Protocol B** (§7).
* **The 50-frame study is not comparable with the literature**, and its window is the first 1.67 seconds of each scene, which may not be representative of the whole.
* **Spacetime Gaussians' cost is the cost of six models.** Its images seen and its steps are sums over six independent trainings, so on a common image budget each of its models has seen a sixth of it.
* **One reader constant is patched** in notebook `05` (§5.1); in the main study it has the upstream value.
* **The initial point cloud may be sparse, and smaller than upstream's**: the shared COLMAP stage asks for `patch_match_stereo` + `stereo_fusion`, which needs a CUDA-enabled COLMAP; where the machine's COLMAP is built without CUDA the stage falls back to the triangulated sparse points and records `init_point_cloud: "sparse"`. Because each method's point budget is a cap and not a target, a sparse fallback means fudan-zvg starts from far fewer than its nominal 300 000 primitives and 4DGaussians from fewer than 40 000. Both the flag and the point count (`init_num_points`) are in every benchmark JSON; runs must not be compared across them.
* **The monitor evaluates with the training renderer.** Spacetime Gaussians' own `test.py` uses a different, fused rasterizer, so the monitor's numbers and those of `test.py` can differ slightly.
* **One run per configuration**, without repeated seeds, as before.
