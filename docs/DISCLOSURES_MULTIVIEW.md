# Disclosures, multi-view study

Things about the multi-view extension that a reader of the results needs to know, and that are **not** open questions: they are settled, they are in the code, and they do not need a decision. They are collected here because each of them would otherwise have to be rediscovered from the notebooks, and because two of them change how a number should be read.

Open questions that *do* need a decision are in [OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md). The full methodology is in [METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md); this page is the short list of what to be careful about.

| # | Disclosure | Affects |
|---|---|---|
| [1](#1-lpips-uses-a-different-backend-from-the-monocular-study) | LPIPS backend differs from the monocular study | comparing LPIPS across the two studies |
| [2](#2-the-segmentation-masks-of-notebook-04-are-an-addition-of-this-project) | Notebook 04's segmentation masks are ours, not the authors' | every result of Dynamic 3D Gaussians |
| [3](#3-the-initial-point-cloud-may-be-smaller-than-upstreams) | The shared point cloud may be smaller than upstream's | 4DGaussians and 4DGS native-4D |
| [4](#4-no-real-time-viewer-in-notebooks-0407) | No real-time viewer in notebooks 04–07 | nothing measured |
| [5](#5-protocol-b-is-not-comparable-for-dynamic-3d-gaussians) | Protocol B truncates the sequence for Dynamic 3D Gaussians | the Protocol B table |
| [6](#6-notebooks-0407-are-generated-not-hand-written) | Notebooks 04–07 are generated, 01–03 hand-written | reproducing the notebooks |
| [7](#7-upstream-code-changed-in-two-notebooks) | Two notebooks patch upstream code, two do not | the "official code unchanged" claim |
| [8](#8-corrections-made-before-the-first-run-2026-09-29) | Two errors corrected before the first run | notebook 05 batch size; the `ours_full` argument |

---

## 1. LPIPS uses a different backend from the monocular study

**What.** The monocular notebooks use each repository's bundled `lpipsPyTorch` (VGG), falling back to the pip `lpips` package only if it is unavailable. The four multi-view notebooks force the **pip `lpips` package** for all of them, with the `[-1, 1]` rescaling it expects, through `LPIPS_BACKEND = "lpips-pip"` in cell 0.1.

**Why.** Dynamic 3D Gaussians does not bundle `lpipsPyTorch` at all. Leaving the backend on `"auto"` would have given three methods the bundled implementation and one the pip package, putting a systematic offset on one row of the LPIPS column — a difference between *measuring instruments*, presented as a difference between methods. Uniformity inside the study was worth more than continuity with the study next door.

**How to read it.** LPIPS is comparable **within** the multi-view study and **within** the monocular study, and should not be compared **between** them. The two implementations are both VGG-backed and both use the official linear weights, so they are close, but they are not the same code and no calibration between them was measured. The resolved backend is written into each run's summary as `lpips_backend`, so a file always says which one produced its numbers.

**Not a problem for:** PSNR, SSIM and the evaluation L1, which are computed by each repository's own modules in both studies, and were checked to be the same formulas (see [METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md) §4.1 #7).

---

## 2. The segmentation masks of notebook 04 are an addition of this project

**What.** Dynamic 3D Gaussians needs a binary dynamic/static mask for every image. Its `seg` loss supervises them directly, and `is_fg` uses them to split the primitives into the ones the rigidity, rotation and isometry losses act on and the background ones that are anchored in place. Panoptic Sports ships those masks. **N3DV does not.**

The notebook estimates them: per camera, the temporal median over the window is taken as the static background, and pixels differing from it by more than `SEG_DIFF_THRESHOLD` (0.08 in mean absolute difference) are called dynamic, after a morphological open/close and a minimum-component-area filter. A point of the initial cloud is labelled foreground when it projects inside the per-camera union-over-time mask in more than half of the cameras that see it.

**Why it matters more than the other adaptations.** The masks are an *input to the method's loss*, not a preprocessing convenience. A mask that is too generous pulls static geometry into the deformable set; one that is too tight anchors moving geometry in place. **The mask quality bounds what this method can achieve here**, and that bound is a property of this preparation, not of the method.

**How to read it.** A low result from notebook 04 is evidence about "Dynamic 3D Gaussians on N3DV with masks derived this way", not about the method in general, and certainly not a refutation of the authors' Panoptic numbers. The thresholds are in every run's benchmark JSON (`seg_diff_threshold`, `seg_min_area_ratio`), and the preparation cell prints the fraction of dynamic pixels per camera — a sanity check worth reading before trusting a run: a scene where that fraction is near 0% or near 100% has a broken mask, not a hard scene.

**Also from the same source:** the authors state that their data-preparation code and their novel-view evaluation code have **not been released**. Both had to be written for this study. The world frame is another consequence: `get_loss` contains a floor loss that assumes a known ground plane, and `get_dataset` hard-codes `near = 1.0`, so the converter infers an orientation, a floor height and a scale that Panoptic simply has. See [METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md) §5.2.

---

## 3. The initial point cloud may be smaller than upstream's

**What.** Notebooks 04, 05 and 06 share one COLMAP reconstruction of the first frame, so that the initialisation is not a hidden difference between the three methods. The stage asks for dense fusion (`patch_match_stereo` + `stereo_fusion`); when the Colab image ships a COLMAP built without CUDA that step cannot run and the stage falls back to the sparse triangulated points, recording `init_point_cloud: "sparse"`.

**Why it matters.** Each method's point budget is a **cap, not a target**. `readNerfSyntheticInfo` subsamples only `if pcd.points.shape[0] > num_pts`; `downsample_point.py` only shrinks a cloud that is too large. Nothing upsamples. So a sparse fallback means 4DGS native-4D starts from far fewer than its nominal 300 000 primitives and 4DGaussians from fewer than 40 000 — a handicap relative to what their authors run, applied to those two methods and not to Spacetime Gaussians, which builds its own per-frame clouds.

**How to read it.** Check `init_point_cloud` and `init_num_points` in the benchmark JSON before comparing two runs, and do not compare across a dense/sparse boundary. The COLMAP stage prints the point count when it builds the cloud, and caches it in `<raw>/<scene>/colmap0/num_points.txt`.

---

## 4. No real-time viewer in notebooks 04–07

**What.** Notebooks 01–03 have a WebSocket streaming viewer (section 6 or 7) that renders the trained model interactively in the Colab output. Notebooks 04–07 do not: their Part 2 is section 5 (rendering and final metrics) and section 6 (MP4 export of the held-out view, side by side with the ground truth).

**Why.** The viewer contributes nothing to any measurement, it is the most repository-specific code in the monocular notebooks, and an N3DV model at 1352×1014 is a poor fit for streaming from a free Colab runtime. Section 5 and section 6 cover everything the benchmark needs: an independent recomputation of the metrics from the saved model, and a visual check of the reconstruction over the whole window.

**Consequence:** none for the results. It is listed only because it is a visible difference in structure between the two families of notebooks, and Part 2 of the multi-view notebooks says so in its own header.

---

## 5. Protocol B is not comparable for Dynamic 3D Gaussians

**What.** Protocol B stops training as soon as the evaluation L1 reaches a per-scene target. For the three methods that optimise the whole window at once, stopping early gives a less converged model *of the whole window* — exactly the intended meaning of "what does this quality cost?".

Dynamic 3D Gaussians walks the sequence frame by frame, so its test curve improves as the window is **covered**, not as one model converges. Stopping early therefore **truncates the sequence**: the saved model represents the frames reached so far and nothing after them.

**How to read it.** `timesteps_done` in the last entry of the run's JSON says how many frames the model actually covers. A Protocol B figure for this method is not comparable with the other three, and **Protocol A is the primary comparison for it**. The machinery works and the run produces a valid file — it just answers a different question.

This is not a defect of the implementation; it is what "equal quality" means for a method whose cost is spent per frame. It is also why the Protocol B targets and the frame-by-frame budget interact: see [OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) §1 and [PROTOCOL_B_CALIBRATION.md](PROTOCOL_B_CALIBRATION.md).

---

## 6. Notebooks 04–07 are generated, not hand-written

**What.** Notebooks 01–03 were written by hand. Notebooks 04–07 are assembled by a small generator, kept **locally only** at `notebooks/_build/` and deliberately excluded from the repository (`.gitignore`).

**Why a generator.** The four multi-view notebooks must share the core of the benchmark monitor *byte for byte* — that is what makes "the same methodology" a checkable property rather than a claim — along with the configuration, path-resolution, dataset, driver and reporting cells. Four hand-maintained copies of a 300-line monitor drift; a generator cannot.

**What it guarantees.** The generator ships with a check suite (246 checks) which verifies, without a GPU or the dataset, that each notebook is valid nbformat-4 JSON, that every code cell compiles as Python, that the generated `benchmark_monitor.py` compiles, **that the monitor core is byte-identical across the four**, that each glue block defines all ten hooks the core calls, that cells 0.1, 0.2 and 3 execute standalone, that a run from a different frame window is never silently reused, and that `_Monitor.step()` produces a well-formed entry against a stubbed repository.

**What it does not guarantee.** Nothing requiring CUDA, COLMAP or the N3DV dataset was executed: the converters, the COLMAP invocations, the rasterizers and every `_render_pair` against a real model are **unverified by execution**. The notebooks are checked as structurally sound and consistent with one another, not as runnable end to end on Colab.

**Consequence for the repository.** Since the generator is not published, the committed `.ipynb` files are the source of truth for anyone else; the byte-identity of the monitor core remains inspectable in them (it is delimited by `# === BENCHMARK MONITOR CORE ===` markers) but is not automatically re-checkable by a third party.

---

## 7. Upstream code changed in two notebooks

The project's rule is that each method runs its **official training code**, modified only by the monitor hook appended to `train.py`, by configuration overrides and by environment fixes. In the multi-view study that rule holds for notebooks 06 and 07 with nothing further. Two notebooks go beyond it, and both changes are inert without the environment variable that drives them:

* **Notebook 05 (4DGaussians)** patches **reader code**: `scene/neural_3D_dataset_NDC.py` hard-codes `countss = 300` and `scene/dataset_readers.py` hard-codes `maxtime=300`; both are made to read `BENCH_N3DV_FRAMES`, defaulting to 300. Without the patch a 50-frame run would timestamp its frames as `idx/300`, compressing the window into the first sixth of the temporal axis of the HexPlane grid while the other three methods span their full range — the run would not be comparable. The other three take the window from a configuration value, so no patch is needed there. Pristine copies are kept as `*.orig`.
* **Notebook 04 (Dynamic 3D Gaussians)** patches **training-loop code**: the constant `10000 if is_initial_timestep else 2000` is made to read two environment variables defaulting to those values, so that the budget knob of [OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) §1 reaches the loop. It also replaces `helpers.o3d_knn` with a `scipy.spatial.cKDTree` implementation returning the same two arrays, because Open3D has no wheel for the Python version Colab ships, and adds a `train_one.py` launcher because `train.py`'s `__main__` hard-codes the six Panoptic Sports sequences.

Everything else — losses and their weights, densification, learning rates, optimisers, stage handling, per-frame initialisation — is upstream in all four notebooks. The full list, with the reason for each, is in [METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md) §5.

---

## 8. Corrections made before the first run (2026-09-29)

Both were found by re-reading the upstream repositories. No run had been made, so no result is affected.

* **Notebook 05 forced `batch_size = 4` on every scene.** The official files `arguments/dynerf/{cook_spinach,cut_roasted_beef,flame_steak,sear_steak}.py` of 4DGaussians set `batch_size=2`, and only `coffee_martini` and `flame_salmon_1` inherit 4 from `default.py`. The derived `<scene>_run.py` overwrote that. This contradicted the rule both studies follow: the per-scene batch is part of a method's tuning and is kept, and declared through `images_seen` ([METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md) §4.2 #1). The override is removed. The batch is now read back from the resolved official config, so `images_seen` and the `batch_size` field of the JSON carry the per-scene value, as notebook 06 already did.
* **The claim that `ours_full` cannot save its decoder was wrong.** The earlier documents said that the released `save_ply()` of Spacetime Gaussians' `oursfull.py` has the decoder save commented out. The commented line is in `ourslite.py`, where there is no decoder to save. In `oursfull.py` the decoder has been written to `point_cloud.pt` since the first commit, and every `load_ply` variant reads it back. Notebook 07's storage report already counted `point_cloud.pt`, and its training renderer already applies the decoder, so `ours_full` works without any change to upstream code. The argument for `ours_lite` that remains is homogeneity alone ([OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) §3).

In the same pass, three settings of cell 0.1 that the documentation tells you to change with `--set` were plain literals, so the command line had no effect on them: `EXTRA_TRAIN_ARGS`, `STG_MODEL` (notebook 07) and `D3DG_BUDGET_MODE`, together with its three companion budget values (notebook 04). All of them now read `BENCH_<NAME>`, and the smoke tests check it. `ours_full` runs also get their own folders (`<scene>_f50_iters30000_full`), so switching the variant can never reuse or overwrite a `lite` run.
