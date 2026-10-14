# Disclosures, multi-view study

Things about the multi-view extension that a reader of the results needs to know, and that are **not** open questions: they are settled, they are in the code, and they do not need a decision. They are collected here because each of them would otherwise have to be rediscovered from the notebooks, and because several of them change how a number should be read.

Choices — taken or still open — are in [OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md). The full methodology is in [METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md); this page is the short list of what to be careful about.

| # | Disclosure | Affects |
|---|---|---|
| [1](#1-protocol-a-is-not-an-equal-budget-comparison) | Protocol A is not an equal-budget comparison | every Protocol A table |
| [2](#2-spacetime-gaussians-at-300-frames-is-one-model-not-the-papers-six) | Spacetime Gaussians at 300 frames is one model, not the paper's six | its comparison with the paper; its VRAM |
| [3](#3-lpips-is-recorded-with-two-backbones-and-the-papers-use-different-ones) | LPIPS is recorded with two backbones, and the papers use different ones | every LPIPS comparison with a paper |
| [4](#4-the-initial-point-cloud-may-be-smaller-than-upstreams) | The shared point cloud may be smaller than upstream's | 4DGaussians and 4DGS native-4D |
| [5](#5-notebooks-0507-are-generated-and-only-smoke-tested) | Notebooks 05–07 are generated, and only smoke-tested | reproducing the notebooks; trusting them before the first run |
| [6](#6-upstream-code-changed-in-two-notebooks) | Two notebooks go beyond the monitor hook | the "official code unchanged" claim |
| [7](#7-corrections-made-before-the-study-runs) | Corrections made before the study runs | what earlier versions of these documents said |
| [8](#8-a-method-was-set-aside-without-being-run) | A method was set aside without being run | what the study does not cover |
| [9](#9-no-real-time-viewer-in-notebooks-0507) | No real-time viewer in notebooks 05–07 | nothing measured |

---

## 1. Protocol A is not an equal-budget comparison

**What.** In the monocular study Protocol A gave every method the same 30 000 steps. Here each method runs at the budget its authors use on N3DV: 3 000 + 14 000 steps for 4DGaussians, 30 000 for 4DGS native-4D, 30 000 for Spacetime Gaussians. The batch sizes are the official ones too, and they differ.

| Method | Steps | Views per step | Images seen in one Protocol A run |
|---|---|---|---|
| 4DGaussians | 17 000 | 2 (four scenes) or 4 (`coffee_martini`, `flame_salmon_1`) | 34 000 or 68 000 |
| 4DGS native-4D | 30 000 | 4 | 120 000 |
| Spacetime Gaussians | 30 000 (one model for the 300 frames) | 2 | 60 000 |

**How to read it.** A Protocol A number says "this method, run as its authors run it". It does not say "this method, given the same resources as the others": the images seen differ by up to a factor of 3.5. A method can be ahead because its authors chose a larger budget. The comparison at equal cost is what Protocol B, and the budget axes of the analysis (images seen, training time), are for.

**The sampling follows the budget.** Each run is sampled about 30 times: every 500 fine steps for 4DGaussians, every 1 000 for the other two. The curves therefore have a comparable number of points but a different spacing in steps.

**What was given up.** The table "at equal steps" of the monocular study has no counterpart. It can be rebuilt for 4DGaussians alone by rerunning notebook 05 with a longer budget ([OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) §6).

---

## 2. Spacetime Gaussians at 300 frames is one model, not the paper's six

**What.** The paper covers a 300-frame sequence with six independent 50-frame models, and its N3DV numbers (32.05 dB for the full model) come from that layout. This study trains **one `ours_full` model on all 300 frames**, a configuration the paper reports on one scene only ([OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) §2). Upstream's code and hyper-parameters are used unchanged, with `--duration 300`.

**How to read it.**

* **Its row is not the paper's configuration.** Its numbers are not to be set next to 32.05 dB. The only published figure of the same configuration is `flame_salmon`: 29.17 dB at 10 000 steps. The pilot run reached 28.82 dB there at 10 000 steps and 28.65 dB at 30 000.
* **It is read at three points**: the end of the 30 000-step run; 10 000 steps, the step count of the paper's 300-frame model (`paper_long_model_iteration`); and the per-scene `test_iteration` of the official configs, which was chosen for 50-frame models.
* **Its peak VRAM does not include the training images.** They are held in host memory as 8-bit integers (`--data_device cpu --gtisint8 1`, upstream options), as the other two methods already do. With the images on the GPU, as upstream does by default, a 50-frame model used 21.1 GB.
* **Only the full variant is run.** There is no `ours_lite` row.

The six-block layout is still in the code (`STG_BLOCK_FRAMES = 50`, the default), with its merge rule: quality is the mean over the 300 test views; training time, steps, images seen and storage are summed over the six models; peak VRAM is their maximum.

---

## 3. LPIPS is recorded with two backbones, and the papers use different ones

**What.** Every evaluation records two LPIPS values: `lpips`, with the VGG backbone, and `lpips_alex`, with AlexNet. Both come from the `lpipsPyTorch` module that each repository bundles, called on images in [0, 1] as the repositories' own evaluation scripts call it.

**Why two.** The papers do not agree. On N3DV, 4DGS native-4D and Spacetime Gaussians report AlexNet; 4DGaussians' code prints both and its paper does not say which one is in its table. The monocular study recorded VGG. ([OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) §9 has the sources.)

**The module is the same in the three repositories**: the four files of `lpipsPyTorch` have identical git blob hashes in all of them (checked on 2026-10-09), so the three methods are scored by the same code, as in the monocular study.

**How to read it.**

* To compare with 4DGS native-4D (0.055) or Spacetime Gaussians (0.044 full, 0.047 lite), use `lpips_alex`.
* For 4DGaussians (0.049) the published backbone is unknown: neither field can be claimed to be the same quantity.
* To compare with the monocular study, use `lpips` (VGG, same module, same call).
* These values are computed on [0, 1] inputs. That is the convention of the whole 3D Gaussian Splatting family, and it is not how the LPIPS reference implementation is normally called, so they should not be set next to LPIPS values produced by other code.
* Each run says which backend produced it (`lpips_backend`). A run made with `--set LPIPS_BACKEND=lpips-pip` (the pip package on inputs rescaled to [-1, 1]) is a different number and must not be mixed with the others.

**Not affected:** PSNR, SSIM and the evaluation L1.

---

## 4. The initial point cloud may be smaller than upstream's

**What.** Notebooks 05 and 06 share one COLMAP reconstruction of the first frame, so that the initialisation is not a hidden difference between the two methods. The stage asks for dense fusion (`patch_match_stereo` + `stereo_fusion`); when the machine's COLMAP is built without CUDA that step cannot run and the stage falls back to the sparse triangulated points, recording `init_point_cloud: "sparse"`.

**Why it matters.** Each method's point budget is a **cap, not a target**. `readNerfSyntheticInfo` subsamples only `if pcd.points.shape[0] > num_pts`; `downsample_point.py` only shrinks a cloud that is too large. Nothing upsamples. So a sparse fallback means 4DGS native-4D starts from far fewer than its nominal 300 000 primitives and 4DGaussians from fewer than 40 000 — a handicap relative to what their authors run, applied to those two methods and not to Spacetime Gaussians, which builds its own per-frame clouds.

**How to read it.** Check `init_point_cloud` and `init_num_points` in the benchmark JSON before comparing two runs, and do not compare across a dense/sparse boundary. The COLMAP stage prints the point count when it builds the cloud, and caches it in `<raw>/<scene>/colmap0/num_points.txt`.

---

## 5. Notebooks 05–07 are generated, and only smoke-tested

**What.** Notebooks 01–03 were written by hand. Notebooks 05–07 are assembled by a small generator, kept **locally only** at `notebooks/_build/` and deliberately excluded from the repository (`.gitignore`).

**Why a generator.** The three multi-view notebooks must share the core of the benchmark monitor *byte for byte* — that is what makes "the same methodology" a checkable property — along with the configuration, path-resolution, dataset, driver and reporting cells. Hand-maintained copies of a 300-line monitor drift; a generator cannot.

**What it guarantees.** The generator ships with a check suite (256 checks) which verifies, without a GPU or the dataset, that each notebook is valid nbformat-4 JSON, that every code cell compiles as Python, that the generated `benchmark_monitor.py` compiles, **that the monitor core is byte-identical across the three**, that each glue block defines all ten hooks the core calls, that the configuration cells execute standalone and honour every command-line override, that each notebook ships its method's official budget and a sampling grid of about 30 points, that the default window is the 300-frame one, that a run from a different frame window is never silently reused, that notebook 07 trains a 300-frame window as six blocks and merges them as defined, and that `_Monitor.step()` produces a well-formed entry against a stubbed repository.

**What it does not guarantee.** The checks execute nothing that needs CUDA, COLMAP or the N3DV dataset: they say the notebooks are structurally sound and consistent with one another. That they run end to end was established separately, by smoke runs of one scene on a cloud GPU on 2026-10-09 (§7), which found and fixed a number of defects the checks could not see. Only one scene was run, with short budgets, so the other five scenes and the full budgets remain untested.

**Consequence for the repository.** Since the generator is not published, the committed `.ipynb` files are the source of truth for anyone else; the byte-identity of the monitor core remains inspectable in them (it is delimited by `# === BENCHMARK MONITOR CORE ===` markers) but is not automatically re-checkable by a third party.

---

## 6. Upstream code changed in two notebooks

The project's rule is that each method runs its **official training code**, modified only by the monitor hook appended to `train.py`, by configuration overrides and by environment fixes. In the multi-view study that rule holds for notebook 06 with nothing further. Two notebooks go beyond it:

* **Notebook 05 (4DGaussians)** patches **reader code**: `scene/neural_3D_dataset_NDC.py` hard-codes `countss = 300` and `scene/dataset_readers.py` hard-codes `maxtime=300`; both are made to read `BENCH_N3DV_FRAMES`, defaulting to 300. **In the main study the value is 300, i.e. the upstream one**, and the patch is inert. It matters for the 50-frame study: without it a 50-frame run would timestamp its frames as `idx/300`, compressing the window into the first sixth of the temporal axis of the HexPlane grid. Pristine copies are kept as `*.orig`.
* **Notebook 07 (Spacetime Gaussians), long model only**, changes **when memory is released**, from the monitor code appended to `train.py` (upstream's files are not edited). Upstream's reader opens every frame as a PIL image and keeps it in the scene description; building the training tensor decodes it, and the decoded copy stays alive next to the tensor. At 300 frames that copy is about 31 GB, and the first attempt was killed at 57 GB on a 62 GB machine. The monitor wraps `loadCamv2` so that each decoded copy is closed once its tensor exists (`BENCH_STG_FREE_DECODED`, set only when one model covers more than 50 frames). The tensors, and therefore what the model trains on, are unchanged; the images then take 24.5 GB of host memory.
* **Notebook 07, rendering (section 5)** runs upstream's `test.py` through a small launcher that translates one call to the current scikit-image: `structural_similarity(..., multichannel=True)` is no longer accepted, and floating-point images now need `data_range`. The launcher passes `channel_axis=-1` and `data_range=1.0`, which is what the old call meant. It affects only `test.py`'s secondary `ssimsv2` value; the benchmark's metrics come from the monitor.

Everything else — losses and their weights, densification, learning rates, optimisers, stage handling — is upstream in all three notebooks. The full list of additions is in [METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md) §5.

---

## 7. Corrections made before the study runs

Found by re-reading the upstream repositories and papers, and by the smoke and pilot runs. None of them affects a result of the study, which has not run yet. They are listed because earlier versions of these documents, in the history of the repository, state the opposite.

### 2026-09-29

* **Notebook 05 forced `batch_size = 4` on every scene.** The official files `arguments/dynerf/{cook_spinach,cut_roasted_beef,flame_steak,sear_steak}.py` of 4DGaussians set `batch_size=2`, and only `coffee_martini` and `flame_salmon_1` inherit 4 from `default.py`. The override is removed; the batch is read back from the resolved official config.
* **The claim that `ours_full` cannot save its decoder was wrong.** The commented-out `torch.save` is in `ourslite.py` (line 391), where there is no decoder to save. In `oursfull.py` the decoder is written to `point_cloud.pt` (line 411) and every `load_ply` variant reads it back. `ours_full` works without any change to upstream code.
* **Three settings that the documentation said to change with `--set` were plain literals**: `EXTRA_TRAIN_ARGS`, `STG_MODEL`, and the budget knobs of the notebook that has since been removed. They now read `BENCH_<NAME>`.

### 2026-10-02 (full re-read of every cited source)

* **Spacetime Gaussians' official test snapshot is per scene.** The documents said its N3DV configs evaluate the 25 000-step snapshot. That is true of four scenes; `coffee_martini` uses 10 000 and `flame_salmon_1` 12 000 (`test_iteration` in `configs/n3d_lite/` and `configs/n3d_full/`). The notebook now records each scene's value and the analysis reads the run there.
* **4DGS native-4D at 300 frames did not use the official time span.** The notebook wrote `time_duration: [0, (N−1)/30]`, i.e. `[0, 9.967]` at 300 frames; the official configs have `[0.0, 10.0]`. It now writes `[0, N/30]`, which is the official value at 300 frames.
* **The GPU of the 4DGaussians paper is stated** (a single RTX 3090, §5.1); the documents said it was not.
* **The 4DGaussians paper and repository disagree** on the batch size (1 in the paper's appendix, 4 or 2 in the N3DV configs) and on the point-cloud cap (100 000 in the paper, 40 000 in the README and the script). The runs follow the repository.
* **The per-frame COLMAP models of Spacetime Gaussians are not reused between its two variants by default**, because a finished scene frees its data. The documents said the stage was shared.
* **The disk estimate for the extracted frames was too high** (15–20 GB for 50 frames). At 2–3 MB per PNG it is roughly 2–3 GB for 50 frames and 13–18 GB for 300. Still an estimate; the runs record the real figure.
* **`--set MAX_ITERATIONS=…` had no effect**: the budget was a literal in cell 0.1. It now reads the environment, which a short smoke run needs.

Everything else that cites a table, a file, a line or an issue was found as cited. Quoted sentences were checked on the README files, the issue threads and the LaTeX sources of the papers.

### 2026-10-09 (first runs on a cloud GPU)

The notebooks were run for the first time, on one scene (`sear_steak`): first on a CPU machine, to find what fails before paying for a GPU, then on an L4. These were short smoke runs, plus one run of 4DGaussians at its full budget. They are not results of the study.

* **The shared COLMAP stage wrote reflected cameras.** Converting the LLFF poses of `poses_bounds.npy` to COLMAP's convention, it negated the middle column of the rotation as well as the last one, so every camera "rotation" had determinant −1. COLMAP then triangulated 357 points with a mean track length of 2.0 (spurious pairs), and dense fusion kept 39. With the conversion that 4DGaussians' `scripts/llff2colmap.py` and Spacetime Gaussians' `script/pre_n3d.py` use — columns (1, 0, −2) — the same 21 images give 3 926 points, a mean track length of 5.8 and a reprojection error of 0.69 px, and those points reproject at a median 0.5 px through the cameras that each of the two readers builds. The stage now asserts that the rotations are proper.
* **What that error cost.** 4DGaussians and 4DGS native-4D started from that cloud. A run of 4DGaussians at its full official budget (3 000 + 14 000 steps, 300 frames) made before the fix reached **25.30 dB** on `sear_steak`, against 32.49 dB in the paper (appendix, Table 6), in the 40 minutes the paper reports. Spacetime Gaussians, which builds its own per-frame clouds with upstream's code, reached 32.7 dB (lite) and 33.5 dB (full) in 2 000 steps. The run was repeated after the fix on 2026-10-13 (below).
* **The 4DGS native-4D conversion was reused across windows.** A scene converted for 50 frames was taken as converted for 300. The conversion now records the window and the cloud it was made from.
* **Environment.** On a fresh lightning.ai Studio the notebooks needed: a CUDA compiler matching PyTorch (the machines ship none on CPU and CUDA 13 on GPU, PyTorch is built for 12.8), `ffmpeg` and `zip`, two standard C++ headers in the CUDA sources, Pillow below 12, `mmcv` built from source for Spacetime Gaussians, COLMAP told to run without a display, and a CUDA build of COLMAP for the dense steps. Each is now done by the notebooks and explained in the cell that does it. Failed installations used to pass silently; the notebooks now stop and name the module.

### 2026-10-13 (pilot runs on an L4)

Two runs at full budget, at 300 frames, before the six scenes: one per open question. They are not results of the study.

* **4DGaussians after the pose fix, `sear_steak`.** Dense fusion now works: 404 259 fused points (39 before the fix), voxel-downsampled to 38 064. The run reached 31.38 dB at the end of the official budget (25.30 before the fix; 32.49 in the paper), with a maximum of 32.93 dB at 3 000 fine steps. Its PSNR oscillates between 30.4 and 32.9 dB from one sample to the next while SSIM and LPIPS improve steadily; that was not investigated. 49 minutes of training, 4.2 GB of VRAM.
* **Spacetime Gaussians, one `ours_full` model on 300 frames, `flame_salmon_1`.** 28.82 dB at 10 000 steps and 28.65 dB at 30 000, against 29.17 dB for the paper's own 300-frame model; slight blur on the fastest-moving parts. Details in [OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) §2. The configuration was accepted on 2026-10-14.
* **Memory of the long model.** The first attempt was killed for lack of host memory; cause and fix in §6.
* **Rendering of notebook 07 failed on a fresh environment**: `test.py` needs scikit-image, which the setup did not install, and calls it in a way current versions reject. The notebook now installs it and runs `test.py` through the launcher of §6. Rendering also asks for the snapshot the run saved (`--test_iteration` = the budget), since the per-scene `test_iteration` of the official config is not saved by a run of the benchmark.
* **Preparation on a CPU machine.** `--set PREPARE_ONLY=true` prepares a scene (frames, COLMAP, conversion) and stops before training, so that the 300 per-frame COLMAP runs of Spacetime Gaussians (75 minutes for `flame_salmon_1` on 16 vCPUs) are not paid at GPU prices.

### 2026-10-09 (check of the LPIPS code)

* **"LPIPS (VGG), as reported in all the papers" was wrong for this dataset.** On N3DV, 4DGS native-4D and Spacetime Gaussians report LPIPS with AlexNet, and 4DGaussians does not state its backbone. The notebooks now record both backbones (§3).
* **The pip `lpips` package with inputs rescaled to [-1, 1] is not what the repositories compute.** They call their bundled `lpipsPyTorch` on images in [0, 1]. The notebooks used the former; they now use the latter, which is the same module in all three repositories.

---

## 8. A method was set aside without being run

Dynamic 3D Gaussians (Luiten et al.) was part of the study, with its own notebook (`04`), until 2026-10-02. It was set aside on the strength of published numbers and of what including it required, **not** of any measurement made here: this project never ran it. The reasons are in [METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md) §1.1 and, with the numbers, in [OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md#dynamic-3d-gaussians-set-aside-2026-10-02).

**How to read it.** The multi-view study says nothing about that method, good or bad. The numbers quoted to justify the choice come from another group's paper, on their protocol and with their tuning. The notebook, which was never executed, remains in the history of the repository.

---

## 9. No real-time viewer in notebooks 05–07

**What.** Notebooks 01–03 have a WebSocket streaming viewer that renders the trained model interactively. Notebooks 05–07 do not: their Part 2 is section 5 (rendering and final metrics) and section 6 (MP4 export of the held-out view, side by side with the ground truth).

**Why.** The viewer contributes nothing to any measurement, it is the most repository-specific code in the monocular notebooks, and an N3DV model at 1352×1014 is a poor fit for streaming from a cloud runtime.

**Consequence:** none for the results. One related gap is real: **rendering speed (FPS) is not measured** by this benchmark, in either study.
