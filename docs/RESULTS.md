# Results: Protocols A and B on the monocular D-NeRF dataset

Analysis of the `benchmark_*.json` files in [`results/`](../results) (3 methods × 8 scenes × 2 protocols, 46 evaluable runs, 1 177 evaluation points). Google Colab **Tesla T4**, 800×800, black background, 20 test views, LPIPS-VGG.

Every number below can be regenerated from the raw JSON with `results/analysis/scripts/run_all.sh`. A subset of the figures backing the claims here is in `results/analysis/figures/` (run the script above to regenerate the full set locally), the tables are in `results/analysis/tables/`, and the interactive dashboard is `results/analysis/benchmark_dashboard.html`.

## 1. Caveats that must accompany these numbers

1. **lego is not evaluable for fudan-zvg**: the run terminates without producing any evaluation entry (the cause was not investigated further). The scene is excluded for that method only.
2. **The fudan-zvg Protocol A runs on trex and jumpingjacks are partial** (6 000 and 15 000 iterations out of 30 000). They had reached their practical limit on a T4: the number of primitives grows to 8.4·10⁵ and 9.8·10⁵, and a thousand iterations cost 1 062 s and 772 s, two orders of magnitude more than the other two methods. The corresponding Protocol B runs are complete.
3. The **best** value of each metric over a run is compared, not the last one: fudan-zvg peaks within the first few thousand iterations and then degrades, and this is also the only comparison the partial runs allow.
4. **An iteration is not the same unit of work for every method**: fudan-zvg uses the per-scene batch of its repository (hellwarrior 1, bouncingballs 2, hook/mutant/standup 8, jumpingjacks 16, trex 24), the other two always use 1. See §4.
5. Aggregate means use the **five scenes completed by all three methods** (bouncingballs, hellwarrior, hook, mutant, standup).

## 2. Protocol A: 30 000 iterations for every method (mean over the 5 common scenes)

| Method | PSNR | SSIM | LPIPS | PSNR peak @ iter | PSNR lost peak→30k | 30k run | s / 1 000 it | peak VRAM | # Gaussians @30k |
|---|---|---|---|---|---|---|---|---|---|
| Deformable-3DGS | **40.76** | **0.991** | **0.014** | 28 800 | 0.01 | 36.7 min | 73 | 3 331 MB | 49 553 |
| 4DGaussians (Wu et al.) | 37.81 | 0.983 | 0.027 | 28 400 ¹ | 0.01 | **18.8 min** | **38** | **1 639 MB** | **40 184** |
| 4DGS native-4D (fudan-zvg) | 34.92 | 0.974 | 0.035 | **10 400** | 0.62 | 113 min | 226 | 2 268 MB | 259 236 |

¹ Total steps, coarse stage included (25 400 counted in the fine stage only; this is the value stored in `runs_summary.csv` as `best_psnr_iter` and shown in the dashboard).

* Deformable-3DGS wins PSNR, SSIM and LPIPS on **all 8 scenes**. 4DGaussians is second in PSNR and SSIM on all scenes, and second in LPIPS on all scenes except mutant and standup, where fudan-zvg has a slightly lower LPIPS (0.0175 vs 0.0209 and 0.0156 vs 0.0161).
* Mean gaps: Deformable-3DGS → 4DGaussians about 3 dB, 4DGaussians → fudan-zvg about 2.9 dB.
* Post-peak degradation of fudan-zvg: −0.62 dB on average, up to −1.26 dB (hook) and −2.14 dB (jumpingjacks). Deformable-3DGS and 4DGaussians are still improving at 30 000 iterations. Meanwhile the fudan-zvg training loss keeps decreasing and its number of primitives multiplies: temporal overfitting, consistent with native 4D primitives having finite temporal support and, in the monocular case, being constrained by very few views each.
* lego is the hard scene for everyone (24.9 / 25.1 dB) and the only one where 4DGaussians and Deformable-3DGS are on par.

## 3. Protocol B: same L1 target, measured at the **first crossing** (mean over the 5 common scenes)

The costs at termination are in the CSV tables but are not the ones to report, because they include the hysteresis delay.

| Method | iterations | images seen | time | PSNR | model on disk | # Gaussians |
|---|---|---|---|---|---|---|
| 4DGS native-4D (fudan-zvg) | **5 258** | 24 598 | 15.97 min | 34.44 | 70.0 MB | 103 237 |
| 4DGaussians (Wu et al.) | 8 315 | **8 315** | **3.49 min** | 33.99 | 17.9 MB | **22 697** |
| Deformable-3DGS | 8 302 | 8 302 | 6.71 min | 34.36 | **12.2 MB** | 40 918 |

All three methods land within 0.5 dB of each other, which is what the protocol imposes; what changes is the price. 4DGaussians is the fastest, at a little more than half the time of Deformable-3DGS and a fifth of fudan-zvg. **The fudan-zvg advantage in iterations (−37%) is an artefact of its larger batches**: counted in training samples it consumes about 3× more, it takes 4.6× the time of 4DGaussians, and its model occupies 3.9× the disk of 4DGaussians (5.7× that of Deformable-3DGS).

## 4. What counts as an equal budget (figure 44)

The ratio iterations / images seen is not an axis: it is exactly `1/batch`, constant within each run. The usable version of the same idea is to **change the independent variable**: not "how many iterations" but "how many samples" or "how many seconds". Three budgets, three answers (mean PSNR over the 5 common scenes). Unlike §2, which reports the *best* value of each run, these are the values *at* the budget, linearly interpolated on the evaluation grid, which is why fudan-zvg reads 34.30 here and 34.92 in §2:

| Budget | 4DGaussians | 4DGS native-4D | Deformable-3DGS |
|---|---|---|---|
| at 30 000 **iterations** | 37.80 | 34.30 | **40.74** |
| at 30 000 **images seen** | 37.80 | 34.37 | **40.74** |
| at equal **wall-clock** time (full run of the fastest method, 756–2 284 s) | 37.80 | 34.08 | **38.36** |

* Moving from iterations to samples **does not change the ranking** and moves fudan-zvg by +0.07 dB: it saturates so early that the extra samples bought by large batches do not become quality. What the sample budget fixes is **coverage**: at 30 000 images the partial trex and jumpingjacks runs are complete (they had done 1 250 and 1 875 steps), so all eight scenes become comparable, with 27.57 dB on trex and 30.89 dB on jumpingjacks for fudan-zvg.
* Moving to wall-clock time **does change the ranking**: the Deformable-3DGS margin shrinks from 2.9 to 0.6 dB, and 4DGaussians wins on hook, mutant and jumpingjacks, because in the time 4DGaussians completes its run Deformable-3DGS has done only part of its own.
* **hellwarrior is the control experiment**: it is the only scene where fudan-zvg also uses batch 1, so the three axes coincide and the ranking is the same under every budget definition.
* Recommended reading: the quality-versus-time curve (figure 08) as the main figure, the iteration budget as the algorithmic comparison, together with the note on per-scene batches. The cost per 1 000 samples is the honest version of throughput.

## 5. Conclusions

* **If maximum quality matters and time is not the constraint: Deformable-3DGS.** It wins on every scene and every metric, by about 3 dB, and at 30 000 iterations it has not yet saturated (its original default is 40 000), so the comparison slightly penalises it. The price is twice the training time and the highest VRAM.
* **If the quality/cost ratio matters: 4DGaussians.** Second in absolute quality, but half the training time of Deformable-3DGS, the lowest VRAM (1.6 GB), the lowest time to a fixed quality target, and first at equal wall-clock time on three of the eight scenes. Its model on disk has a fixed component due to the HexPlane grid, independent of the number of primitives, so on simple scenes it can occupy more than Deformable-3DGS with fewer primitives.
* **4DGS with native 4D primitives** converges in very few optimisation steps but on this dataset is the most expensive in time, model size and number of primitives, and it is the only method that degrades when trained past its optimum. On the high-batch scenes it cannot even complete 30 000 iterations on a T4. The limit appears structural rather than implementational: native 4D primitives are penalised by the monocular setting, where each instant offers a single view, and the authors state that they introduce no architectural prior for this case.
* **The practical constraint is time, not memory.** Peak VRAM never exceeded 5.1 GB of the 16 GB available.

## 6. Declared limitations

See [METHODOLOGY.md](METHODOLOGY.md) §9. In short: lego is excluded for fudan-zvg; two fudan-zvg Protocol A runs are partial; the uniform budget misaligns the upstream schedules (tuned for 20 000, 20 000 and 40 000 steps); Gaussian counts depend on densification hyper-parameters; one run per configuration without repeated seeds (fluctuations of about 0.1 dB); training times also reflect the quality of each CUDA implementation.

## 7. Extending the benchmark

To add a method (for example Spacetime Gaussians) or a scene:

1. Copy its output folder next to the others in `results/`, with the same `<run>/benchmark/benchmark_*.json` layout.
2. `results/analysis/scripts/aggregate.py`: add the folder to `METHOD_DIR` and the label to `SHORT`.
3. `results/analysis/scripts/common.py`: add the short tag to `C` (colour, from the validated palette), `ORDER` and `LABEL`; mark partial runs in `PARTIAL` and non-evaluable method/scene pairs in `NOT_EVAL`. Every figure handles both automatically (dashed lines, hatching, "n/a").
4. Update `FAIR` (scenes completed by every method) in `fig_agg.py`, `fig_analysis.py` and `fig_fair.py`.
5. Run `./run_all.sh`.
