#!/usr/bin/env python3
"""Turn a few short smoke runs into a GPU-hours estimate for the multi-view study.

A smoke run is an ordinary Protocol A run of one scene, stopped early with
`--set MAX_ITERATIONS=<n>` and written to its own folder with `--set ROOT_NAME=<name>`,
so that it can never be mistaken for a result:

    python3 scripts/run_benchmark.py notebooks/05_4dgaussians_wu_n3dv.ipynb \
        --set RUN_MODE=single --set SCENE=sear_steak --set NUM_FRAMES=50 \
        --set MAX_ITERATIONS=2000 --set EVAL_EVERY_N_ITERS=500 \
        --set ROOT_NAME=dgs-smoke --set STORAGE_MODE=local

This script reads every benchmark JSON under the smoke folder and prints two tables:

  1. what was measured: seconds per step, seconds per evaluation, peak VRAM, preparation
     time and disk of the scene;
  2. what the full study would cost, per method and per window, for six scenes.

    python3 scripts/estimate_gpu_hours.py <workdir>/bench_out/dgs-smoke

How the estimate is built, and what it cannot know:

  * training = seconds per step x the method's official budget. The rate is taken from
    the last sampling interval of the smoke run. It is a LOWER BOUND: a short run stops
    while the number of Gaussians is still growing (densification runs until step 9 000,
    10 000 or 15 000 depending on the method), and a step gets slower as they grow. The
    number of Gaussians at the end of the smoke run is printed so this can be judged.
  * evaluation = seconds per evaluation x the number of samples of a full run. This is not
    training time, but it is GPU time that is paid for.
  * preparation = the measured preparation time of the scene (download, frame extraction,
    COLMAP, conversion), near zero if the scene was already cached when the run started.
  * a window that was not measured is not guessed, with one exception: Spacetime Gaussians
    at 300 frames is six independent 50-frame models, so its training and its per-frame
    COLMAP are six times the 50-frame figures.

Standard library only.
"""

import glob
import json
import os
import sys

N_SCENES = 6
# Spacetime Gaussians has no `iterations` in its N3DV configs: 30 000 is the default of
# thirdparty/gaussian_splatting/arguments/__init__.py.
STG_OFFICIAL_ITERATIONS = 30000
STG_BLOCK_FRAMES = 50


def load_runs(root):
    runs = []
    pattern = os.path.join(root, "*", "*", "benchmark", "benchmark_iterations.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            summary = json.load(handle)
        entries = summary.get("entries") or []
        if len(entries) < 2:
            print("skipped (fewer than two samples): %s" % path, file=sys.stderr)
            continue
        runs.append((path, summary, entries))
    return runs


def describe(path, summary, entries):
    cfg = summary.get("config") or {}
    storage = summary.get("storage") or {}
    variant = cfg.get("stg_model") or storage.get("stg_model")
    method = summary.get("method", "?")
    if variant:
        method = "%s [%s]" % (method, variant)
    frames = int(summary.get("num_frames") or cfg.get("num_frames") or 0)
    blocks = 1
    if variant:
        blocks = int(cfg.get("stg_num_blocks") or max(1, frames // STG_BLOCK_FRAMES))

    last, prev = entries[-1], entries[-2]
    steps = last["total_iterations"] - prev["total_iterations"]
    seconds = last["training_time_s"] - prev["training_time_s"]
    tail_rate = seconds / steps if steps > 0 else float("nan")
    mean_rate = last["training_time_s"] / max(last["total_iterations"], 1)

    evals = [e.get("eval_duration_s") for e in entries if e.get("eval_duration_s")]
    eval_s = sum(evals) / len(evals) if evals else float("nan")

    if variant:
        official_per_model = STG_OFFICIAL_ITERATIONS
    else:
        official_per_model = (int(cfg.get("official_iterations") or cfg.get("max_iterations"))
                              + int(cfg.get("official_coarse_iterations") or 0))
    every = int(cfg.get("eval_every_n_iters") or 1000)
    return {
        "path": path,
        "method": method,
        "is_stg": bool(variant),
        "scene": summary.get("scene"),
        "frames": frames,
        "blocks": blocks,
        "steps_run": last["total_iterations"],
        "tail_rate": tail_rate,                 # s per step, last interval
        "mean_rate": mean_rate,                 # s per step, whole smoke run
        "eval_s": eval_s,                       # s per evaluation (whole test split)
        "eval_views": last.get("num_eval_views"),
        "gaussians": last.get("num_gaussians"),
        "vram_mb": summary.get("peak_vram_mb_nvidia_smi"),
        "prep_s": summary.get("preparation_time_s"),
        "data_mb": summary.get("scene_data_mb"),
        "official_steps": official_per_model,   # per model (per block for STG)
        "smoke_every": every,
    }


def hours(seconds):
    return seconds / 3600.0


def estimate(run, frames, samples_per_model):
    """GPU-machine hours for six scenes of one method on one window, or None."""
    models = 1
    scale = 1.0
    if run["frames"] != frames:
        if not (run["is_stg"] and run["frames"] == STG_BLOCK_FRAMES and frames % STG_BLOCK_FRAMES == 0):
            return None
        models = frames // STG_BLOCK_FRAMES
        scale = float(models)
    elif run["is_stg"]:
        models = max(run["blocks"], 1)
    per_model_steps = run["official_steps"]
    # A smoke run in blocks reports summed steps and time, so its rate is already per step.
    train = run["tail_rate"] * per_model_steps * models
    evaluation = (run["eval_s"] / max(run["blocks"], 1)) * samples_per_model * models
    prep = (run["prep_s"] or 0.0) * scale
    disk = (run["data_mb"] or 0.0) * scale
    return {
        "train_h": hours(train) * N_SCENES,
        "eval_h": hours(evaluation) * N_SCENES,
        "prep_h": hours(prep) * N_SCENES,
        "total_h": hours(train + evaluation + prep) * N_SCENES,
        "disk_gb_scene": disk / 1024.0,
        "extrapolated": run["frames"] != frames,
    }


def fmt(value, pattern="%.2f"):
    if value is None or value != value:
        return "n/a"
    return pattern % value


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    runs = [describe(*r) for r in load_runs(sys.argv[1])]
    if not runs:
        sys.exit("no smoke run found under %s" % sys.argv[1])

    print("## Measured (one scene, smoke run)\n")
    print("| method | scene | frames | steps run | s/step (last interval) | s/step (mean) | "
          "s/evaluation | test views | Gaussians at stop | peak VRAM (MB) | preparation (min) | "
          "scene data (GB) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        print("| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["method"], r["scene"], r["frames"], r["steps_run"],
            fmt(r["tail_rate"], "%.3f"), fmt(r["mean_rate"], "%.3f"), fmt(r["eval_s"], "%.1f"),
            r["eval_views"], r["gaussians"], fmt(r["vram_mb"], "%.0f"),
            fmt(None if r["prep_s"] is None else r["prep_s"] / 60.0, "%.1f"),
            fmt(None if r["data_mb"] is None else r["data_mb"] / 1024.0, "%.1f")))

    print("\n## Estimated cost of Protocol A, six scenes, at the official budget\n")
    print("Training hours are a lower bound (see the header of this script).\n")
    print("| method | window | official steps per model | training (h) | evaluation (h) | "
          "preparation (h) | total GPU-machine time (h) | disk per scene (GB) | basis |")
    print("|---|---|---|---|---|---|---|---|---|")
    for frames in (300, 50):
        for r in runs:
            # About 30 samples per model, plus the baseline one, as the notebooks ship.
            samples = 31 if r["official_steps"] >= 30000 else 29
            e = estimate(r, frames, samples)
            if e is None:
                continue
            basis = ("6 x the %d-frame smoke run" % r["frames"]) if e["extrapolated"] \
                else ("%d-frame smoke run" % r["frames"])
            print("| %s | %d frames | %d | %s | %s | %s | %s | %s | %s |" % (
                r["method"], frames, r["official_steps"], fmt(e["train_h"]), fmt(e["eval_h"]),
                fmt(e["prep_h"]), fmt(e["total_h"]), fmt(e["disk_gb_scene"], "%.1f"), basis))
    print("\nProtocol B (50 frames) stops when the target is reached, normally well before the "
          "official budget, so the 50-frame rows are a rough upper bound for it too.")
    print("A method with no row for a window was not measured on it: run its smoke test with "
          "that NUM_FRAMES.")


if __name__ == "__main__":
    main()
