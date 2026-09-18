# Monocular D-NeRF benchmark: analysis of the raw results

Generated from the `benchmark_*.json` files produced by the three training pipelines.
All runs: **Google Colab, NVIDIA Tesla T4**, D-NeRF synthetic monocular dataset, 8 scenes,
evaluation at 800x800 on 20 held-out views, black background, LPIPS with the VGG backbone.

## 1. Methods and experimental protocols

| Tag | Method | Folder | Schedule |
|---|---|---|---|
| 4DGaussians | 4DGaussians / HexPlane (Wu et al.) | `4dgaussian_output` | 3 000 coarse + 27 000 fine = **30 000 total**, batch 1 |
| 4DGS-fudan | 4D Gaussian Splatting, native 4D primitives (fudan-zvg) | `4dgs_fudan_output` | single stage, 30 000 iterations, **per-scene batch 1-24** |
| Deformable-3DGS | Deformable-3D-Gaussians (Yang et al.) | `deformablegaussian` | single stage, 30 000 iterations, batch 1 |

* **Protocol A — equal iteration budget** (`*_iters*` runs): the same 30 000 total optimisation
  steps for every method; the **best** value of each metric over the run is reported. The 3 000
  coarse iterations of 4DGaussians are counted, so the step budget is identical.
* **Protocol B — equal loss budget** (`*_loss*` runs): training stops once the evaluation **L1**
  reaches a common per-scene target (2 consecutive hits), so the methods are compared **at the same
  reconstruction fidelity** and what varies is the cost of getting there. Targets follow the
  calibration rule `target = 1.05 x max_methods(min L1 on the test curve)`. All 23 runs reached
  their target. **Every Protocol-B cost reported here is measured at the first crossing of the
  target, not at run termination.**

Per-scene L1 targets: bouncingballs 0.005083 · hellwarrior 0.005358 · hook 0.006253 ·
jumpingjacks 0.004610 · lego 0.013466 · mutant 0.003106 · standup 0.002168 · trex 0.005793.

**Batch sizes matter.** 4DGS-fudan uses the per-scene batch of its own repository —
hellwarrior 1, bouncingballs 2, hook/mutant/standup 8, jumpingjacks 16, trex 24 — while the other
two methods always use 1. An "iteration" is therefore not the same unit of work across methods:
the batch-size-corrected axis is **training images seen**.

## 2. Caveats (must be quoted with any of these figures)

1. **Lego is not evaluable for 4DGS-fudan** — the run terminates without producing any evaluation
   entry (the cause was not investigated further).
   Lego is excluded from every 4DGS-fudan number.
2. **T-Rex and Jumping Jacks Protocol-A runs of 4DGS-fudan are partial** (stopped at 6 000 and
   15 000 iterations). They had already reached their practical limit on a T4: the primitive count
   grows to 8.4x10^5 and 9.8x10^5 and a thousand iterations costs 1 062 s and 772 s — two orders of
   magnitude above the other methods. Their Protocol-B runs are complete and reached the target.
   In every figure these runs are drawn with dashed lines / hatched bars.
3. Because 4DGS-fudan peaks within the first few thousand iterations and then degrades, comparing
   the **best** metric of each run (rather than the final one) is the fair choice; it is also what
   the partial runs allow.
4. Aggregate means use the **five scenes completed by all three methods** (bouncingballs,
   hellwarrior, hook, mutant, standup).

## 3. Headline results

### Protocol A — equal iteration budget (mean over the 5 common scenes)

| Method | PSNR (dB) | SSIM | LPIPS | peak PSNR at iter. | PSNR lost peak->30k | 30k run (min) | s / 1 000 iters | peak VRAM (MB) | # Gaussians @30k |
|---|---|---|---|---|---|---|---|---|---|
| Deformable-3DGS | **40.76** | **0.991** | **0.014** | 28 800 | 0.01 | 36.7 | 73.4 | 3 331 | 49 553 |
| 4DGaussians | 37.81 | 0.983 | 0.027 | 28 400 ¹ | 0.01 | **18.8** | **37.6** | **1 639** | **40 184** |
| 4DGS-fudan | 34.92 | 0.974 | 0.035 | **10 400** | 0.62 | 113.0 | 226.0 | 2 268 | 259 236 |

¹ Total steps including the 3 000 coarse iterations; `best_psnr_iter` in `runs_summary.csv` counts the fine stage only (25 400).

Deformable-3DGS has the best PSNR, SSIM and LPIPS on **all 8 scenes**; 4DGaussians is second in PSNR and SSIM on all
of them (in LPIPS 4DGS-fudan is marginally better on mutant and standup) and is the cheapest per iteration and in memory. The mean PSNR gap Deformable -> 4DGaussians
is about 3 dB, and 4DGaussians -> 4DGS-fudan about 2.9 dB.

### Protocol B — equal L1 target, measured at the first crossing (mean over the 5 common scenes)

| Method | iterations | training images seen | time (min) | PSNR (dB) | model on disk (MB) | # Gaussians |
|---|---|---|---|---|---|---|
| 4DGS-fudan | **5 258** | 24 598 | 15.97 | 34.44 | 70.0 | 103 237 |
| 4DGaussians | 8 315 | **8 315** | **3.49** | 33.99 | 17.9 | **22 697** |
| Deformable-3DGS | 8 302 | 8 302 | 6.71 | 34.36 | **12.2** | 40 918 |

All three land within 0.5 dB of each other, as the protocol intends; the interesting quantity is
the price. 4DGS-fudan appears to need ~37 % fewer optimisation steps, but that advantage is an
artefact of its larger batches: **in training samples it consumes about 3x more**, it takes
**4.6x more wall-clock time than 4DGaussians** and a model **3.9x larger than 4DGaussians' (5.7x Deformable-3DGS')**, because the native-4D
formulation densifies to several times more primitives.

### The 4DGS-fudan peak

4DGS-fudan reaches its best PSNR at iteration 4 000-16 000 (mean 10 400 on the common scenes) and
then decays: -0.62 dB on average, up to -1.26 dB (hook) and -2.14 dB (jumping jacks) by the end of
the run, while the other two methods are still improving monotonically at 30 000 iterations. This
is the behaviour that motivates Protocol B and the use of the best-of-run value.

## 4. What counts as an equal budget

Because 4DGS-fudan trains with per-scene batches of 1 to 24, a step is not a unit of work that can be
compared across methods, and three different budgets give three different answers (figure 44):

| Budget (mean over the 5 common scenes) | 4DGaussians | 4DGS-fudan | Deformable-3DGS |
|---|---|---|---|
| PSNR after 30 000 **optimisation steps** | 37.80 | 34.30 | **40.74** |
| PSNR after 30 000 **training images** | 37.80 | 34.37 | **40.74** |
| PSNR at the same **wall-clock** time (the fastest method's full run, 756-2 284 s) | 37.80 | 34.08 | **38.36** |

* Switching from steps to samples barely moves 4DGS-fudan (+0.07 dB on average): it plateaus so early
  that the extra samples its larger batches consume buy almost nothing. What the sample budget *does*
  fix is coverage — **at 30 000 images the partial T-Rex and Jumping Jacks runs are complete** (they had
  run 1 250 and 1 875 steps at that point), so all eight scenes become comparable: T-Rex 27.57 dB and
  Jumping Jacks 30.89 dB for 4DGS-fudan.
* Switching to wall-clock changes the ranking: Deformable-3DGS loses 2.4 dB and **4DGaussians wins on
  hook, mutant and jumping jacks**, because within the time Wu needs for a full run Deformable has only
  completed part of its own.
* Hell Warrior is the natural control experiment: it is the only scene where 4DGS-fudan also uses batch
  1, so steps, samples and iterations coincide for all three methods, and the ranking there is the same
  under all three budgets.

## 5. Contents

```
analysis/
  README_analysis.md          this file
  benchmark_dashboard.html    interactive dashboard (self-contained, opens offline; its charts are
                               drawn by Chart.js from data inlined in the page, independent of the
                               figures/ folder below)
  dashboard_data.json         data behind the dashboard
  scripts/                    reproducible pipeline: ./run_all.sh regenerates every table, figure and
                               the dashboard from the raw JSON in ../
  figures/                    the 7 figures below; run_all.sh regenerates the full set of 46 locally
  tables/                     curves_all.csv, runs_summary.csv, table_iso_iterations.csv,
                              table_iso_loss.csv, table_aggregates_fair5.csv
```

Only the figures that back a specific claim made in this repository's prose (`README.md`,
`RESULTS.md`, above) are kept committed here, to keep the repository focused on the results rather
than on every intermediate view of them. The full 46-figure set is one command away: run
`scripts/run_all.sh` against this repository (it reads only the raw JSON already committed under
`results/`) to regenerate all of it, dashboard included, bit-for-bit.

| Figure | Content | Backs |
|---|---|---|
| `08_psnr_vs_traintime` | PSNR vs training time | the quality/cost curve embedded in `README.md` |
| `17_iso_loss_time_to_target` | Protocol B: minutes to the common L1 target | "4DGaussians... the lowest time to a fixed quality target" |
| `22_peak_vram_iters` | peak VRAM per method | "the practical constraint is time, not memory" |
| `27_peak_and_degradation` | iteration of peak PSNR and quality lost afterwards | the 4DGS-fudan post-peak degradation finding |
| `30_quality_vs_cost_scatter` | quality vs compute, both protocols in one view | the overall quality/cost picture |
| `31_aggregate_quality_dots` | aggregate PSNR/SSIM/LPIPS dot plot | the ~3 dB Deformable-3DGS lead |
| `44_winner_by_budget_definition` | winner under each definition of "equal budget" | the budget-definition table above and the embedded figure in `README.md` |
