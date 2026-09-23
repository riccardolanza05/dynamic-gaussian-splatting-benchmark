# Open decisions, multi-view study

Three choices in the multi-view extension are **judgement calls, not facts**. Each is implemented with a default and a switch, so none of them blocks a run; but each changes what the resulting numbers mean, and the default was chosen by the person who wrote the notebooks, not by the person who owns the study.

This page exists so the decision can be taken later, with the trade-off in front of you, rather than discovered when the tables are already built. Each entry says what is implemented today, what the alternative is, what it costs to switch, and what the number means under each option.

Status of all three: **default in place, not yet decided.** Nothing here has to be settled before running Protocol A; but #2 is expensive to change afterwards, so read it first.

| # | Decision | Current default | Cost of changing later |
|---|---|---|---|
| [1](#1-iteration-budget-of-dynamic-3d-gaussians) | Protocol A budget of Dynamic 3D Gaussians | `"native"` | one knob, rerun that notebook |
| [2](#2-temporal-window-50-frames) | Temporal window | 50 frames | **everything must be rerun** |
| [3](#3-which-spacetime-gaussians-model) | Spacetime Gaussians variant | `ours_lite` | one knob, rerun that notebook |

---

## 1. Iteration budget of Dynamic 3D Gaussians

**Where:** cell 0.1 of [`notebooks/04_dynamic3dgaussians_luiten_n3dv.ipynb`](../notebooks/04_dynamic3dgaussians_luiten_n3dv.ipynb), the variable `D3DG_BUDGET_MODE`.

### The problem

The other three methods run one optimisation over the whole window, so "30 000 optimisation steps" is a budget that means the same thing for all of them. Dynamic 3D Gaussians does not work that way: it reconstructs the first frame from scratch, then walks the sequence frame by frame, spending a fixed number of steps on each. Its total is a *function of the window*, not a number anyone chose:

```
total = ITERS_FIRST + ITERS_PER_TIMESTEP × (NUM_FRAMES − 1)
```

With the authors' values (10 000 and 2 000) and a 50-frame window that is **108 000 steps**, 3.6× the budget of the other three. There is no setting of that method which both equals 30 000 total steps and leaves its per-frame budget alone.

### Option A — `"native"` (implemented)

`D3DG_ITERS_FIRST = 10000`, `D3DG_ITERS_PER_TIMESTEP = 2000`: the authors' schedule, unchanged.

* **What the number means:** what this method does when you run it as its authors intend. Comparable with their own results in kind, though not in value (see decision 2).
* **What it costs:** the equal-iteration axis of Protocol A stops being equal for this method. A "Protocol A" table would put 108 000 steps next to 30 000 and the step column would be misleading on its own.
* **Why it was chosen as the default:** this is the precedent the monocular study already set. It did not equalise the per-scene `batch_size` of 4DGS native-4D (1 to 24 views per step), because doing so would have reported a number about a method nobody runs; instead it declared the difference and analysed it on the `images_seen` axis (see [METHODOLOGY.md](METHODOLOGY.md) §3.5 and [RESULTS.md](RESULTS.md) §4). The same argument applies here, more strongly: the per-frame budget is not a tuning knob but the shape of the algorithm.
* **Wall-clock consequence:** 108 000 steps at batch 1 on a T4. This is the longest run of the four notebooks by a wide margin, and the one most likely to hit a Colab session limit. The loop is resumable per scene, but not mid-scene.

### Option B — `"aligned"`

`D3DG_BUDGET_MODE = "aligned"` keeps the first frame at 10 000 and divides the remaining `ALIGNED_TOTAL_ITERATIONS − 10 000` equally over the later frames. At 30 000 total and 50 frames that is **408 steps per frame** instead of 2 000.

* **What the number means:** what this method does at a budget its authors never proposed. 408 steps is roughly a fifth of the tracking budget per frame; the persistence losses (rigidity, rotation, isometry, background anchoring) have that much less opportunity to settle before the window moves on. The result is very likely to be a floor, not a measurement of the method.
* **What it buys:** a Protocol A table where the iteration column is genuinely one budget for four methods, so "at equal steps, which reconstructs best?" has a literal answer.
* **What it costs:** the answer is arguably about the budget, not about the representation. It also tells you nothing about the method's practical cost, which is what Protocol B is for.

### Option C — run both

The knob makes this cheap in code and expensive in GPU time: one extra full loop of notebook 04. The run folders already carry the budget in their name (`<scene>_f50_iters108000` against `<scene>_f50_iters30000`), so the two coexist without collision, and `budget_mode` is recorded in every benchmark JSON. Reporting native as the headline and aligned as a footnote is the most defensible outcome, if the GPU time is available.

### What to weigh

If the point of the multi-view study is *"what do these four methods do on multi-camera video"*, take **A**. If the point is *"at one fixed budget, which representation wins"*, take **B** and say plainly in the results that this method is being run outside its design point. If the study is meant to stand as a reference, **C**.

**Recommendation:** A, with C if there is GPU budget. It matches what the monocular study already did with an unalignable unit of work.

---

## 2. Temporal window: 50 frames

**Where:** cell 0.1 of all four notebooks, the variable `NUM_FRAMES`. **This is the one decision that is expensive to revisit**, because it invalidates every run made under the other value.

### Why 50 and not 300

N3DV sequences are 300 frames at 30 fps. Three constraints, each of which alone would force a shorter window on a free-tier T4:

* **Spacetime Gaussians covers a sequence in chunks of 50 frames by construction** (`duration: 50` in every official N3DV config). At 300 frames it is six independent models per scene: six trainings, six sets of metrics to merge and six times the storage. The benchmark has no defined way to attribute one PSNR to six models.
* **Dynamic 3D Gaussians costs a fixed number of steps per frame** (decision 1). At 300 frames the native schedule is **608 000 steps** per scene. On a T4 that is not a long run, it is an impossible one.
* **Disk.** One scene at 1352×1014, 21 cameras × 50 frames is already 15–20 GB of extracted PNG; 300 frames is six times that on a runtime that also has to hold the model and the checkpoints.

Fifty frames is the longest window where all four methods fit the same hardware *and* where Spacetime Gaussians' native chunk equals the whole window, so no method is penalised by chunking. It also makes the test split exactly 50 views for every method, which is what lets `MAX_EVAL_VIEWS` stay at `0` and keeps the evaluation L1 directly comparable, exactly as in the monocular study.

### What it costs

**No published N3DV number is comparable with these runs.** Every paper in the comparison reports the 300-frame sequence. The check that [METHODOLOGY.md](METHODOLOGY.md) §7 performs for the monocular study — measured PSNR against published PSNR, scene by scene, which is what caught the background convention issue — **has no counterpart in the multi-view study**. The multi-view results are internally comparable and externally they are not, and that has to be said wherever they are quoted.

This is not a small loss. In the monocular study that check is what turned "our numbers look plausible" into "our numbers agree with the papers to within 0.37 dB, and where they do not, here is why".

### Option A — keep 50 frames everywhere (implemented)

Four methods, one window, one protocol, no external validation.

### Option B — 50 frames for all four, plus a 300-frame run of notebooks 05 and 06 only

The two methods that can do 300 frames on a T4 are 4DGaussians and 4DGS native-4D: both take the window from a configuration value and neither cost scales with it the way the other two do. A secondary pair of runs at `NUM_FRAMES = 300` would restore the literature check for those two, and by extension give some confidence that the 50-frame numbers of the other two are not distorted by the preparation.

* **What it buys:** the ability to say "at 300 frames our 4DGaussians matches the published value to within X dB, so the pipeline is sound; the 50-frame table below is then internally comparable".
* **What it costs:** two more full loops, six scenes each, at six times the frames. Disk is the binding constraint, not GPU: `DELETE_FRAMES_AFTER_TRAIN` already frees each scene after its run, so it is feasible one scene at a time. The run folders separate cleanly (`_f300_` against `_f50_`) and `run_is_complete()` refuses to mix windows, so nothing can be confused.
* **What it does not buy:** any comparison between the 300-frame and 50-frame runs. They are different experiments.

### Option C — 300 frames for everything

Not feasible for Dynamic 3D Gaussians on this hardware, and awkward for Spacetime Gaussians (six models per scene). Would require dropping one or both methods, which is the thing this extension exists to avoid.

### What to weigh

The question is whether the study is allowed to have **no external validation at all**. Option A accepts that; option B buys it back for half the methods at the cost of two more loops.

**Recommendation:** B, if the disk and the time are there. The literature check is the single most valuable thing the monocular study has, and losing it entirely is a real weakening of the extension. If only A is possible, the limitation is already written into [METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md) §2.3 and §8 and must stay attached to every number.

---

## 3. Which Spacetime Gaussians model

**Where:** cell 0.1 of [`notebooks/07_spacetime_gaussians_li_n3dv.ipynb`](../notebooks/07_spacetime_gaussians_li_n3dv.ipynb), the variable `STG_MODEL`.

### The two released models

The repository ships two, with separate configs (`configs/n3d_lite/` and `configs/n3d_full/`) and separate rasterizers:

* **`ours_lite`** — 3-channel direct RGB. The primitive carries its colour; the rasterizer (`diff_gaussian_rasterization_ch3`) writes it out.
* **`ours_full`** — 9-channel features decoded to RGB by a small MLP (`diff_gaussian_rasterization_ch9` + `rgbdecoder`). Higher quality in the paper, and the variant its headline table reports.

### Option A — `ours_lite` (implemented)

* **Homogeneity.** None of the other three methods in this comparison has a neural colour decoder: 4DGaussians and 4DGS native-4D use spherical harmonics, Dynamic 3D Gaussians a per-primitive RGB. `ours_full` would introduce a learned per-model component the others do not have, which lands in the storage column, the VRAM column and the training-time column at once, and is hard to attribute.
* **A concrete blocker.** In the released `save_ply()` of `oursfull.py` the line that writes the decoder is **commented out**:

  ```python
  model_fname = path.replace(".ply", ".pt")
  print(f'Saving model checkpoint to: {model_fname}')
  # torch.save(self.rgbdecoder.state_dict(), model_fname)
  ```

  So an `ours_full` model saved by upstream code **cannot be reloaded for rendering**, and its storage figure would be wrong — the benchmark would charge it for the point cloud and not for the decoder it also needs. Benchmarking it honestly means adding a save step upstream does not have, which is exactly the kind of change the rest of the project avoids.
* **What it costs:** the paper's headline N3DV row is `Ours`, not `Ours-lite`. A reader who knows the paper will expect the higher number. The gap in the paper's own N3DV table is modest (about 0.5 dB) but it is real.

### Option B — `ours_full`

* **What it buys:** the variant the paper leads with, so the method is represented at its best.
* **What it requires:** (a) building `gaussian_rasterization_ch9` instead of `ch3` — the notebook already selects the right one from `STG_MODEL`, so this is automatic; (b) **an added save of the decoder**, which does not exist upstream, plus its bytes added to `model_storage_report`; (c) declaring in the methodology that one of the four methods has a neural colour decoder and three do not.
* **What it costs:** more VRAM and more time per step on a T4, and one more upstream deviation on the list.

### What to weigh

Whether the multi-view study wants **the methods as their authors present them** (B) or **the methods reduced to a common shape so the columns mean the same thing** (A). The monocular study leaned consistently towards the second: it forced one resolution, one background, one budget, one LPIPS backbone.

**Recommendation:** A. The unsaved-decoder bug is the deciding argument — with `ours_full` the storage column, which is one of the ten monitored metrics, would be measuring the wrong thing unless the notebook adds a save the authors did not write.

---

## How to record the decision

When one is taken, do all three of:

1. set the knob in cell 0.1 of the affected notebook;
2. note the choice and the date in this file, replacing the entry's "not yet decided";
3. if the choice differs from what [METHODOLOGY_MULTIVIEW.md](METHODOLOGY_MULTIVIEW.md) §4 and §6 describe, update those sections too — they state the current defaults as fact.

Every choice is already recorded per run in the benchmark JSON (`budget_mode`, `num_frames`, `stg_model`), so a run always carries the decision it was made under, whatever this file says.
