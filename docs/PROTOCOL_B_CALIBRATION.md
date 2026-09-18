# Protocol B: calibration of the evaluation-loss (L1) targets

Derived from the Protocol A `benchmark_iterations.json` files of the three methods (800×800, black background, 20 test views, LPIPS-VGG, Tesla T4).

## Rule

    target(scene) = 1.05 × max over methods ( minimum eval_l1_loss on the method's test curve )

The **minimum of the curve** (the value at the peak) is used, not the last sample. For Deformable-3DGS and 4DGaussians the two agree within 2%, because their curves are still descending at the end of the budget. For fudan-zvg the last sample is degraded by the temporal overfitting that follows the peak on high-batch scenes, and using it would set the bar at a worse model than the method can reach. The rule is applied uniformly to the three methods.

The 5% margin matters: without it the slowest method would meet the target exactly where its curve flattens, and the two-consecutive-hits rule might never confirm, ending the run on the safety cap.

`lego` is the only scene where the binding method is not fudan-zvg: the fudan-zvg run on lego terminates immediately without producing any evaluation entry (the cause was not investigated further), so the scene is **excluded for fudan-zvg only** and the target is set by 4DGaussians.

Note: cell 4.3 of each notebook prints candidates from the *final* eval L1 of each Protocol A run. The values below apply the rule above to the curve minimum instead, computed from the same JSON files.

## Values (identical in the three notebooks)

| scene | Deformable-3DGS L1 min | 4DGaussians L1 min | fudan-zvg L1 min | binding | target (×1.05) |
|---|---|---|---|---|---|
| bouncingballs | 0.002063 @29k | 0.002843 @13k | 0.004841 @18k | fudan-zvg | **0.005083** |
| hellwarrior | 0.001894 @29k | 0.002581 @27k | 0.005103 @10k | fudan-zvg | **0.005358** |
| hook | 0.003120 @28k | 0.004303 @27k | 0.005956 @4k | fudan-zvg | **0.006253** |
| jumpingjacks | 0.002152 @30k | 0.002414 @25k | 0.004391 @4k | fudan-zvg | **0.004610** |
| lego | 0.012148 @15k | 0.012825 @10k | excluded | 4DGaussians | **0.013466** |
| mutant | 0.001653 @29k | 0.002410 @26k | 0.002958 @11k | fudan-zvg | **0.003106** |
| standup | 0.001234 @30k | 0.001633 @27k | 0.002065 @10k | fudan-zvg | **0.002168** |
| trex | 0.002157 @30k | 0.003803 @24k | 0.005517 @4k | fudan-zvg | **0.005793** |

Partial fudan-zvg runs used for the calibration: `jumpingjacks` stopped at 15 000 / 30 000 and `trex` at 6 000 / 30 000 because of Colab limits (batch sizes 16 and 24 make training very slow). In both cases the peak had been passed well before the interruption, so the curve minimum is genuine.

## Expected first crossing (from the Protocol A curves, 1 000-iteration grid)

| scene | Deformable-3DGS | 4DGaussians | fudan-zvg |
|---|---|---|---|
| bouncingballs | 8 000 / 484 s | 3 000 / 226 s | 8 000 / 853 s |
| hellwarrior | 5 000 / 119 s | 2 000 / 84 s | 5 000 / 228 s |
| hook | 8 000 / 393 s | 6 000 / 221 s | 3 000 / 903 s |
| jumpingjacks | 11 000 / 435 s | 5 000 / 186 s | 3 000 / 1 656 s |
| lego | 8 000 / 868 s | 6 000 / 325 s | n/a |
| mutant | 10 000 / 687 s | 10 000 / 352 s | 5 000 / 1 434 s |
| standup | 14 000 / 567 s | 9 000 / 294 s | 5 000 / 1 368 s |
| trex | 10 000 / 789 s | 7 000 / 305 s | 4 000 / 3 753 s |

Indicative totals for the full loop, before hysteresis: Deformable-3DGS about 1.2 h, 4DGaussians about 33 min, fudan-zvg about 2.8 h. No scene crosses its target before `MIN_ITERATIONS_BEFORE_STOP = 1000`, so that guard never binds and does not bias the times.

## Protocol B parameters (set in the three notebooks)

`TRAINING_MODE = "target_eval_loss"`, `EVAL_EVERY_N_MINUTES = 0.5`, `MIN_ITERATIONS_BEFORE_STOP = 1000`, `TARGET_CONSECUTIVE_HITS = 2`, `SAFETY_MAX_ITERATIONS = 60000`, `TARGET_METRIC = "eval_loss"`, `EVAL_LOSS_KIND = "l1"`, `MAX_EVAL_VIEWS = 0`, `KEEP_MODEL_ON_DRIVE = False`.

The notebooks ship with `TRAINING_MODE = "iterations"` (Protocol A); switch it to `"target_eval_loss"` to run Protocol B. The time to report is the **first crossing** of the target, not the termination; cell 4.4 extracts it.
