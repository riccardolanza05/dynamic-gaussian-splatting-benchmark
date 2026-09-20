# Methodology

This document describes how the benchmark was built: which methods were compared, how the measurements were taken without altering the official training loops, which conventions had to be aligned across the three repositories, and why. Results are in [RESULTS.md](RESULTS.md); the Protocol B targets are derived in [PROTOCOL_B_CALIBRATION.md](PROTOCOL_B_CALIBRATION.md).

## 1. Scope

**Goal.** A controlled comparison of dynamic 3D Gaussian Splatting methods on dynamic scenes, with a uniform measurement protocol checked against the published numbers.

**Methods benchmarked** (the three that accept monocular input):

| Method | Repository | Representation |
|---|---|---|
| Deformable-3DGS (Yang et al., CVPR 2024) | [`ingra14m/Deformable-3D-Gaussians`](https://github.com/ingra14m/Deformable-3D-Gaussians) | canonical space + per-Gaussian MLP deformation field |
| 4DGaussians / HexPlane (Wu et al., CVPR 2024) | [`hustvl/4DGaussians`](https://github.com/hustvl/4DGaussians) | canonical space + factorised HexPlane grid + light decoder |
| 4DGS with native 4D primitives (Yang et al., ICLR 2024) | [`fudan-zvg/4d-gaussian-splatting`](https://github.com/fudan-zvg/4d-gaussian-splatting) | native 4D Gaussians with finite temporal extent |

Not benchmarked: Dynamic 3D Gaussians (Luiten et al.), which requires multi-view input by construction, and Spacetime Gaussian Feature Splatting (Li et al.). Both were studied as part of the literature review ([REFERENCES.md](REFERENCES.md), [papers_comparison_table.pdf](papers_comparison_table.pdf)).

**Dataset.** D-NeRF synthetic, monocular, 8 scenes (`bouncingballs`, `hellwarrior`, `hook`, `jumpingjacks`, `lego`, `mutant`, `standup`, `trex`), native 800×800 RGBA images, 20 test views per scene.

**Hardware.** Google Colab free tier, NVIDIA Tesla T4 (16 GB).

The three representations differ in a way that explains most of the results. In the two canonical-space methods every primitive exists at every instant, so every training image constrains all primitives. In the native-4D method a primitive is constrained only by the views that fall inside its temporal support; in the monocular setting, where each instant has a single view, the problem is much less constrained.

## 2. Instrumentation

### 2.1 The injected monitor

The central problem was measuring metrics *during* training without changing upstream behaviour. Each repository calls its own `training_report(...)` at every iteration of the training loop. The notebooks append a few lines to `train.py`, just before `if __name__ == "__main__":`, that wrap that function with a monitor:

```python
import os as _bench_os
if _bench_os.environ.get('BENCH_CONFIG'):
    import benchmark_monitor as _bench
    training_report = _bench.make_hook(training_report)
```

This works because Python resolves global names at run time: `training()`, defined earlier, calls the wrapped version. A pristine copy is kept as `train.py.orig`, and without the `BENCH_CONFIG` environment variable the file behaves exactly as upstream. The wrapper recovers the loop state (iteration, scene, render function, deformation module, stage) through `inspect.signature().bind()`, so it is robust to signature differences between repositories.

### 2.2 Metrics recorded at every sample

Computed with **each repository's own modules**: `utils.image_utils.psnr`, `utils.loss_utils.ssim`, `utils.loss_utils.l1_loss`, and the bundled `lpipsPyTorch` (VGG backbone), with the pip `lpips` package as fallback. `l1_loss` and `ssim` are identical in the three repositories; `psnr` is the same formula (4DGaussians adds an optional mask argument, unused here); the bundled `lpipsPyTorch` package is identical up to blank lines.

Fields of every JSON entry: `iteration`, `total_iterations`, `images_seen`, `stage`, `training_time_s`, `wall_time_s`, `benchmark_overhead_s`, `psnr`, `ssim`, `lpips`, `eval_l1_loss`, `eval_photometric_loss`, `eval_loss`, `eval_loss_kind`, `num_eval_views`, `eval_resolution`, `num_gaussians`, `train_batch_loss`, `eval_duration_s`.

Summary fields: `status`, `target_reached`, `best_psnr`, `best_psnr_iteration`, `process_wall_time_s`, `peak_vram_mb_nvidia_smi`, `storage`.

### 2.3 Time measurement

`training_time_s` is wall-clock time **net of the benchmark overhead**: evaluation, JSON writing and saving the best checkpoint. An early version saved fudan-zvg's `chkpnt_best.pth` *after* the timed window, so the write was charged to training time (an estimated 5–10% inflation); the overhead window now covers it.

Peak VRAM is measured by polling `nvidia-smi` from a separate thread every 2 s, because training runs in a subprocess and `torch.cuda.max_memory_allocated()` in the notebook kernel would measure the kernel itself.

## 3. Aligned conventions and how they were reached

### 3.1 Resolution: 800×800

The fudan-zvg YAML sets `resolution: 2`, which halves the images: fudan-zvg was evaluated at 400×400 while the other two ran at 800×800 (default `_resolution = -1`, native). The clue was the evaluation overhead, 1.35 s against 4.7 s, a ~3.5× ratio consistent with a quarter of the pixels. `resolution` is forced to 1 in fudan-zvg, and `eval_resolution` is recorded in every JSON entry as a permanent guarantee. A footnote of the fudan-zvg paper confirms 800×800 rendering for its published table ("otherwise downsampled 2× by default").

### 3.2 Background: black, imposed by fudan-zvg

In `fudan-zvg/scene/cameras.py` the ground truth is **premultiplied by the alpha channel**:

```python
if gt_alpha_mask is not None:
    self.image *= gt_alpha_mask.to(self.image.device)
```

D-NeRF images are RGBA with a transparent background, so fudan-zvg's ground truth is **always black**, whatever `white_background` says. That flag, however, also sets the rasterizer background: setting it to `True` makes the model render on white against a black ground truth, and **training collapses** (PSNR from 23 to 11.8, SSIM from 0.94 to 0.79). The other two methods are unaffected because their `dataset_readers.py` produce 3-channel `"RGB"` images, so `loaded_mask` stays `None` and the premultiplication never happens.

Black is therefore used for all three: Deformable-3DGS at its repository default, Wu et al. with `ModelParams = dict(white_background = False)` in the derived config, fudan-zvg at its default.

Measured effect of the background on Deformable-3DGS / bouncingballs: 43.16 dB on white against 40.64 dB on black, **+2.52 dB on white**. Background pixels are trivial to reconstruct and inflate PSNR; LPIPS is almost insensitive. The Deformable-3DGS paper explicitly reports D-NeRF experiments on a black background at full 800×800 resolution, matching this convention.

### 3.3 Iteration budget: 30 000 steps in total

Original D-NeRF defaults: Wu et al. 20 000 fine + 3 000 coarse; fudan-zvg 20 000 (in the scene YAML; the class default is 30 000); Deformable-3DGS 40 000.

Deformable-3DGS has `warm_up = 3000` **inside** `--iterations` (deformation forced to zero); Wu et al. have `coarse_iterations = 3000` **on top**. The two phases are conceptually equivalent: both optimise static Gaussians. The budget is therefore `MAX_ITERATIONS = 30000` for Deformable-3DGS and fudan-zvg and **27000 for Wu et al.**, which aligns the methods on both axes:

| | static warm-up | dynamic optimisation | total |
|---|---|---|---|
| Deformable-3DGS | 3 000 | 27 000 | 30 000 |
| 4DGaussians (Wu et al.) | 3 000 (coarse) | 27 000 (fine) | 30 000 |
| 4DGS native-4D (fudan-zvg) | none | 30 000 | 30 000 |

fudan-zvg has no warm-up on the D-NeRF path: the paper mentions one in the section on urban scenes, and it is not implemented here.

**Schedules are not rescaled.** Learning rates and densification stay at the upstream values, tuned for different budgets. At 30 000 steps Deformable-3DGS is *below* its optimum (default 40 000) and still improving; Wu et al. and fudan-zvg are *beyond* theirs (default 20 000). This is a deliberate choice and is declared as a limitation.

### 3.4 Official per-scene configuration for Wu et al.

`train.py` loads the config with `mmcv.Config.fromfile`, but recent `mmcv` moved `Config` into `mmengine`. Without `--configs`, the method ran with generic hyper-parameters its authors never proposed for D-NeRF. The notebook writes a `mmcv.py` shim in the repository root (it wins on `sys.path` because the script directory comes first) that implements `Config.fromfile` with recursive `_base_` resolution and deep merge.

`merge_hparams` applies the config **over** the parsed arguments, so file values override the command line, `iterations` included. Budget and background are therefore written into a derived `<scene>_run.py`.

Differences from the generic defaults: `multires [1,2]` instead of `[1,2,4,8]`, `defor_depth 0` instead of 1, `deformation_lr_final` and `grid_lr_final` ten times lower, `pruning_interval 8000` instead of 100, `weight_decay_iteration 0` instead of 5000, plus a per-scene `kplanes_config`. Across scenes only the fourth (temporal) component of the k-planes resolution changes:

| temporal resolution | scenes |
|---|---|
| 25 | lego |
| 50 | hellwarrior, hook |
| 75 | bouncingballs, mutant, standup |
| 100 | jumpingjacks, trex |

Everything else is inherited unchanged from `dnerf_default.py`.

### 3.5 The fudan-zvg batch size

The fudan-zvg YAML configs vary strongly by scene:

| scene | batch_size | num_pts |
|---|---|---|
| hellwarrior | 1 | 100 000 |
| bouncingballs, lego | 2 | 100 000 |
| hook, mutant, standup | 8 | 300 000 |
| jumpingjacks | 16 | 300 000 |
| trex | 24 | 300 000 |

The code implements **gradient accumulation**: an inner loop renders `batch_size` views, divides each loss by `batch_size` and calls `backward()` without zeroing the gradients, then performs a single `optimizer.step()`. The views have different timestamps, so each step sees N instants of the sequence.

The uniform budget fixes **optimiser steps**, not images seen: on `trex` fudan-zvg sees 720 000 images against 30 000 for the other two. This is recorded in the `images_seen` column and analysed in [RESULTS.md](RESULTS.md) §4. The fudan-zvg paper states a default batch size of 4, which matches none of the released configs. All eight scenes use `gaussian_dim: 4`, `rot_4d: True`, `force_sh_3d: False`.

### 3.6 Other alignments

* `--test_iterations 0` disables the native periodic evaluation, which would otherwise double the overhead. For fudan-zvg, `exhaust_test: False` (upstream `True`, which evaluates the whole test split every 500 iterations); the monitor maintains `chkpnt_best.pth` with the same rule on its own schedule.
* `MAX_EVAL_VIEWS = 0` (all 20 views), required for the L1 values to be comparable.
* LPIPS with the VGG backbone. Deformable-3DGS and fudan-zvg state it explicitly in their papers. Wu et al. do not specify it, but their `metrics.py` computes both; the VGG value (0.0158 measured) is very close to the published one (0.0155), whereas AlexNet gives 0.0071.

## 4. Definition of the evaluation loss

**Choice.** Mean L1 on the test split, used both as the Protocol B stopping criterion and as the main loss column.

**Rationale, checked against the papers and the code.** L1 is the common denominator of all methods. 3DGS and Deformable-3DGS optimise `(1-0.2)·L1 + 0.2·D-SSIM`; Wu et al. optimise `‖Î − I‖ + L_tv` with `lambda_dssim = 0` (no SSIM term); fudan-zvg uses `λ_l1·L1 + λ_dssim·L_dssim`. It is also exactly what the three repositories accumulate as `l1_test` in their own logs.

**What is excluded and why.** The extra terms (grid TV, time smoothness, rigidity, motion, sparsity) are functions of the model parameters alone. They never look at the ground truth, take the same value on train and test, and have **no evaluation counterpart**.

**PSNR as an alternative.** In the repositories PSNR is `20·log10(1/√MSE)`, a monotonic transform of the L2 error: a pixel-wise error criterion like L1, only L2-based and logarithmic. Its advantage is external comparability, since papers publish PSNR and none publishes test L1. `eval_photometric_loss` with λ = 0.2 is recorded in parallel as a diagnostic.

## 5. Model storage

### 5.1 Method

Determined **empirically** by leave-one-out: removing one file at a time from a copy of the run and re-running `render.py` and `metrics.py`.

| method | required files | confirmed optional |
|---|---|---|
| Deformable-3DGS | `cfg_args`, `point_cloud/iteration_*/point_cloud.ply`, `deform/iteration_*/deform.pth` | `cameras.json`, `input.ply`, TensorBoard events |
| 4DGaussians | `cfg_args`, `point_cloud.ply`, `deformation.pth` | `deformation_table.pth`, `deformation_accum.pth`, events |
| 4DGS native-4D | `cfg_args`, `chkpnt_best.pth` | `chkpnt<N>.pth`, `cameras.json`, `input.ply`, events |

For Wu et al., removing `deformation_table.pth` leaves the metrics **bit-identical** (PSNR 37.06026077270508 in both cases).

### 5.2 Splitting the fudan-zvg checkpoint

`capture()` for `gaussian_dim == 4` returns 19 elements:

```
0 active_sh_degree   1 _xyz          2 _features_dc   3 _features_rest
4 _scaling           5 _rotation     6 _opacity       7 max_radii2D
8 xyz_gradient_accum 9 t_gradient_accum  10 denom     11 optimizer
12 spatial_lr_scale  13 _t           14 _scaling_t    15 _rotation_r
16 rot_4d            17 env_map      18 active_sh_degree_t
```

Model elements: **1, 2, 3, 4, 5, 6, 13, 14, 15**. The Adam state (element 11) and the densification accumulators (7, 8, 9, 10), about two thirds of the file, are excluded. Validated end to end on a `hellwarrior` development run: 185.79 MB on disk, 61.42 MB of model tensors, 27 bytes of rounding difference from the JSON. In every fudan-zvg run in `results/` the model tensors are about one third of the checkpoint. Element 3 (`_features_rest`, 47 harmonic coefficients per channel) alone is 90% of the model.

### 5.3 Residual asymmetry

For Deformable-3DGS and Wu et al. the figure is **bytes on disk**; for fudan-zvg it is **bytes of in-memory tensors**. The difference is negligible but it is a difference of definition.

## 6. Protocols

**Protocol A, fixed budget.** `MAX_ITERATIONS` (30 000, or 27 000 fine iterations for Wu et al.), metrics sampled every `EVAL_EVERY_N_ITERS = 1000`.

**Protocol B, fixed quality target.** Training stops when the evaluation L1 reaches `TARGET_EVAL_LOSS_PER_SCENE[scene]`; metrics sampled every `EVAL_EVERY_N_MINUTES = 0.5`. Parameters: `MIN_ITERATIONS_BEFORE_STOP = 1000`, `TARGET_CONSECUTIVE_HITS = 2`, `SAFETY_MAX_ITERATIONS = 60000`, `TARGET_METRIC = "eval_loss"`. The targets are identical in the three notebooks; their calibration is described in [PROTOCOL_B_CALIBRATION.md](PROTOCOL_B_CALIBRATION.md).

**Hysteresis** is necessary. On fudan-zvg / bouncingballs the sequence around the threshold was 0.005186 (below), 0.005206, 0.005208 (above), 0.005192, 0.005127 (below), then stop. Without hysteresis the run would have stopped on a fluctuation.

**The time to report is the first crossing**, not the termination. The hysteresis delay differs by method: on an early bouncingballs run it added 70% for Wu et al., 17% for Deformable-3DGS and 27% for fudan-zvg. Cell 4.4 of each notebook extracts the first crossing automatically, and the analysis uses the `fc_*` columns of `runs_summary.csv`.

`MIN_ITERATIONS_BEFORE_STOP` is compared with `iteration`, which for Wu et al. is the fine-stage iteration: the guard therefore triggers at 1 000 absolute steps for two methods and at 4 000 for Wu et al. No scene crosses its target before that point, so the guard never binds.

## 7. Validation against the literature

* **Deformable-3DGS, bouncingballs:** 40.64 dB measured against 41.01 published; SSIM 0.9957 against 0.9953; LPIPS 0.0090 against 0.0093. 0.37 dB lower PSNR with slightly better SSIM and LPIPS, at 30 000 instead of 40 000 iterations, under identical conditions (800×800, black background, LPIPS-VGG).
* **4DGaussians, bouncingballs:** 37.08 dB against 40.62 published, but **LPIPS agrees within 3%** (0.0160 against 0.0155). The PSNR gap is explained by the background convention: Wu et al. default to white, and the measured effect of white is +2.52 dB.
* **4DGS native-4D:** the paper reports only the eight-scene mean (34.09 PSNR, 0.98 SSIM, 0.02 LPIPS) with no per-scene table, so it cannot be checked scene by scene.

The ranking Deformable-3DGS > 4DGaussians > 4DGS native-4D matches the papers' own averages (39.51, 34.05, 34.09).

## 8. Other findings from the implementation work

* **Temporal overfitting of fudan-zvg correlates with batch size.** On high-batch scenes the test curve peaks early and then declines monotonically while the training loss keeps falling and the number of Gaussians multiplies (e.g. jumpingjacks, batch 16: PSNR peak 32.25 dB at iteration 5 000 with 3.2·10⁵ Gaussians, then 30.11 dB and 9.8·10⁵ Gaussians at iteration 15 000). A gradient averaged over many views has lower variance, which lets densification proceed unchecked; native 4D primitives have finite temporal support, so in the monocular case the new primitives are constrained by very few views and memorise instead of generalising. The authors state that they introduce no architectural prior specific to the monocular case.
* **Wu et al. storage has a fixed component.** `deformation.pth` is 11.07 MB on hellwarrior (7 259 Gaussians) and 10.83 MB on hook (28 927 Gaussians), both with temporal resolution 50, against 12.91 MB on bouncingballs (resolution 75). The HexPlane grid costs the same regardless of the number of primitives, so on hellwarrior Wu et al. occupy twice the disk of Deformable-3DGS with less than half the primitives.
* **Gaussian counts are driven by hyper-parameters.** Wu et al. go from 31 269 to 120 372 Gaussians on bouncingballs just by changing `pruning_interval` from 100 to 8000 (generic defaults against official config), with unchanged LPIPS. The count measures a hyper-parameter choice, not a property of the representation.
* **Deformable-3DGS instability.** On bouncingballs in Protocol B, at iteration 6 034 the L1 jumps from 0.0056 to 0.0104 and PSNR drops from 29.9 to 26.1 before recovering: this is `opacity_reset_interval = 3000`. The best method in final quality is also the least stable during optimisation.
* **Differences between sources.** Wu et al.'s `metrics.py` gives 37.060 dB against 37.079 from the monitor, because it computes on 8-bit PNGs instead of float tensors. Deformable-3DGS's `metrics.py` passes `[0,1]` images to the pip `lpips` package, which expects `[-1,1]` with `normalize=False`; the monitor uses `lpipsPyTorch` and rescales correctly in the pip fallback, so its values do not match that script exactly. `train_batch_loss` is the loss of the **last view of the batch**, not the mean, so for fudan-zvg it is noisier and not comparable across methods; the fudan-zvg progress bar shows training PSNR (up to 48 dB on jumpingjacks), not test PSNR.

## 9. Declared limitations

* The uniform budget misaligns each method's schedules, tuned for different original budgets (20 000, 20 000 and 40 000 steps).
* fudan-zvg uses its per-scene `batch_size` (1 to 24), so a step is not the same unit of work across methods (see [RESULTS.md](RESULTS.md) §4).
* The number of Gaussians depends strongly on the densification hyper-parameters and is not an intrinsic property of the representation.
* Training times also reflect the implementation quality of each CUDA rasterizer, not only the algorithmic cost.
* One run per configuration, without repeated seeds; observed fluctuations are of the order of 0.1 dB.
* The set of files needed for storage was verified on one scene only; the storage definition is asymmetric for fudan-zvg (§5.3).
* `lego` is the scene for which the Deformable-3DGS authors report a mismatch between training and test splits and use the validation set as test; the notebooks do not do this.
* `lego` cannot be evaluated for fudan-zvg (its run terminates without any evaluation entry; not investigated further), and the fudan-zvg Protocol A runs on `trex` and `jumpingjacks` are partial (see [RESULTS.md](RESULTS.md) §1).
