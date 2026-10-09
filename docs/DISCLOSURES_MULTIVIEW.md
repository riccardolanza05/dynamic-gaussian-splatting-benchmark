# Disclosures, multi-view study

Things about the multi-view extension that a reader of the results needs to know, and that are **not** open questions: they are settled, they are in the code, and they do not need a decision. They are collected here because each of them would otherwise have to be rediscovered from the notebooks, and because several of them change how a number should be read.

Choices — taken or still open — are in [OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md). The full methodology is in [METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md); this page is the short list of what to be careful about.

| # | Disclosure | Affects |
|---|---|---|
| [1](#1-protocol-a-is-not-an-equal-budget-comparison) | Protocol A is not an equal-budget comparison | every Protocol A table |
| [2](#2-spacetime-gaussians-at-300-frames-is-six-models) | Spacetime Gaussians at 300 frames is six models | its time, steps, images and storage in the main study |
| [3](#3-lpips-is-recorded-with-two-backbones-and-the-papers-use-different-ones) | LPIPS is recorded with two backbones, and the papers use different ones | every LPIPS comparison with a paper |
| [4](#4-the-initial-point-cloud-may-be-smaller-than-upstreams) | The shared point cloud may be smaller than upstream's | 4DGaussians and 4DGS native-4D |
| [5](#5-notebooks-0507-are-generated-and-nothing-has-run-on-a-gpu) | Notebooks 05–07 are generated, and nothing has run on a GPU | reproducing the notebooks; trusting them before the first run |
| [6](#6-upstream-code-changed-in-one-notebook) | One notebook patches upstream reader code | the "official code unchanged" claim |
| [7](#7-corrections-made-before-the-first-run) | Corrections made before the first run | what earlier versions of these documents said |
| [8](#8-a-method-was-set-aside-without-being-run) | A method was set aside without being run | what the study does not cover |
| [9](#9-no-real-time-viewer-in-notebooks-0507) | No real-time viewer in notebooks 05–07 | nothing measured |

---

## 1. Protocol A is not an equal-budget comparison

**What.** In the monocular study Protocol A gave every method the same 30 000 steps. Here each method runs at the budget its authors use on N3DV: 3 000 + 14 000 steps for 4DGaussians, 30 000 for 4DGS native-4D, 30 000 for Spacetime Gaussians. The batch sizes are the official ones too, and they differ.

| Method | Steps | Views per step | Images seen in one Protocol A run |
|---|---|---|---|
| 4DGaussians | 17 000 | 2 (four scenes) or 4 (`coffee_martini`, `flame_salmon_1`) | 34 000 or 68 000 |
| 4DGS native-4D | 30 000 | 4 | 120 000 |
| Spacetime Gaussians | 30 000 per model | 2 | 60 000 per model |

**How to read it.** A Protocol A number says "this method, run as its authors run it". It does not say "this method, given the same resources as the others": the images seen differ by up to a factor of 3.5. A method can be ahead because its authors chose a larger budget. The comparison at equal cost is what Protocol B, and the budget axes of the analysis (images seen, training time), are for.

**The sampling follows the budget.** Each run is sampled about 30 times: every 500 fine steps for 4DGaussians, every 1 000 for the other two. The curves therefore have a comparable number of points but a different spacing in steps.

**What was given up.** The table "at equal steps" of the monocular study has no counterpart. It can be rebuilt for 4DGaussians alone by rerunning notebook 05 with a longer budget ([OPEN_DECISIONS_MULTIVIEW.md](OPEN_DECISIONS_MULTIVIEW.md) §6).

---

## 2. Spacetime Gaussians at 300 frames is six models

**What.** The method covers a sequence in blocks of 50 frames. In the main study a scene is six independent trainings, as in its paper, and the benchmark JSON of the scene is their merge: quality is the mean over the 300 test views; training time, steps, images seen, number of Gaussians and storage are **sums** over the six; peak VRAM is the maximum.

**How to read it.**

* Its training time and storage are those of everything needed to render the 300 frames, which is the fair figure to set next to a method that needs one model.
* Its `total_iterations` is 180 000 and its `images_seen` 360 000, but no single model has trained for more than 30 000 steps. On a "same number of images" comparison each of its models has seen a sixth of the common budget, so that comparison understates it at 300 frames; the 50-frame study, where it is one model, is the place to read it.
* **It is read at two points.** Its official flow evaluates one snapshot per scene (`test_iteration`: 25 000 on four scenes, 10 000 on `coffee_martini`, 12 000 on `flame_salmon_1`), not the end of the 30 000-step run. Both readings are reported; the snapshot is the one comparable with the paper.
* **It appears as up to two rows**, `lite` and `full`. They are two separate sets of runs of the same notebook.

The per-block JSON files are kept under `<run>/blocks/`, so every merged number can be recomputed.

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

## 5. Notebooks 05–07 are generated, and nothing has run on a GPU

**What.** Notebooks 01–03 were written by hand. Notebooks 05–07 are assembled by a small generator, kept **locally only** at `notebooks/_build/` and deliberately excluded from the repository (`.gitignore`).

**Why a generator.** The three multi-view notebooks must share the core of the benchmark monitor *byte for byte* — that is what makes "the same methodology" a checkable property — along with the configuration, path-resolution, dataset, driver and reporting cells. Hand-maintained copies of a 300-line monitor drift; a generator cannot.

**What it guarantees.** The generator ships with a check suite (245 checks) which verifies, without a GPU or the dataset, that each notebook is valid nbformat-4 JSON, that every code cell compiles as Python, that the generated `benchmark_monitor.py` compiles, **that the monitor core is byte-identical across the three**, that each glue block defines all ten hooks the core calls, that the configuration cells execute standalone and honour every command-line override, that each notebook ships its method's official budget and a sampling grid of about 30 points, that the default window is the 300-frame one, that a run from a different frame window is never silently reused, that notebook 07 trains a 300-frame window as six blocks and merges them as defined, and that `_Monitor.step()` produces a well-formed entry against a stubbed repository.

**What it does not guarantee.** Nothing requiring CUDA, COLMAP or the N3DV dataset was executed: the converters, the COLMAP invocations, the rasterizers and every `_render_pair` against a real model are **unverified by execution**. The notebooks are checked as structurally sound and consistent with one another, not as runnable end to end. The first smoke run is the first real test.

**Consequence for the repository.** Since the generator is not published, the committed `.ipynb` files are the source of truth for anyone else; the byte-identity of the monitor core remains inspectable in them (it is delimited by `# === BENCHMARK MONITOR CORE ===` markers) but is not automatically re-checkable by a third party.

---

## 6. Upstream code changed in one notebook

The project's rule is that each method runs its **official training code**, modified only by the monitor hook appended to `train.py`, by configuration overrides and by environment fixes. In the multi-view study that rule holds for notebooks 06 and 07 with nothing further. One notebook goes beyond it:

* **Notebook 05 (4DGaussians)** patches **reader code**: `scene/neural_3D_dataset_NDC.py` hard-codes `countss = 300` and `scene/dataset_readers.py` hard-codes `maxtime=300`; both are made to read `BENCH_N3DV_FRAMES`, defaulting to 300. **In the main study the value is 300, i.e. the upstream one**, and the patch is inert. It matters for the 50-frame study: without it a 50-frame run would timestamp its frames as `idx/300`, compressing the window into the first sixth of the temporal axis of the HexPlane grid. Pristine copies are kept as `*.orig`.

Everything else — losses and their weights, densification, learning rates, optimisers, stage handling — is upstream in all three notebooks. The full list of additions is in [METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md) §5.

---

## 7. Corrections made before the first run

All were found by re-reading the upstream repositories and papers. No run had been made, so no result is affected. They are listed because earlier versions of these documents, in the history of the repository, state the opposite.

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
