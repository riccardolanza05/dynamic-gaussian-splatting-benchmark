# Analysis scripts

Regenerate every table, figure and the dashboard from the raw `benchmark_*.json` files.

```bash
./run_all.sh
```

Requires `python3` with `numpy`, `pandas`, `matplotlib`. No network access needed
(except to view the CDN-linked dashboard; see `build_dashboard.py --inline-lib`).

## Order of execution

| Script | Reads | Writes |
|---|---|---|
| `aggregate.py` | all `*/benchmark/benchmark_*.json` | `tables/curves_all.csv`, `tables/runs_summary.csv` |
| `tables.py` | those two CSVs | `tables/table_iso_iterations.csv`, `table_iso_loss.csv`, `table_aggregates_fair5.csv` |
| `fig_curves.py` | `curves_all.csv` | figures 01–10 |
| `fig_compare.py` | both CSVs | figures 11–24, 38 |
| `fig_analysis.py` | both CSVs | figures 25–30 |
| `fig_agg.py` | both CSVs | figures 31–35 |
| `fig_extra.py` | both CSVs | figures 36–37 |
| `fig_fair.py` | both CSVs | figures 39–44 (equal-budget comparisons) |
| `export_web.py` | both CSVs | `dashboard_data.json` |
| `build_dashboard.py` | `dash_template.html` + that JSON | `benchmark_dashboard.html` |

**Note on the dashboard.** The `benchmark_dashboard.html` currently in `analysis/` has Chart.js
embedded, so it opens with no network. `run_all.sh` rebuilds it with a CDN `<script>` tag instead,
which needs internet the first time it is opened. To get a fully offline copy back, fetch
`chart.umd.js` (Chart.js 4.4.4, e.g. `npm pack chart.js@4.4.4` or the cdnjs URL in
`dash_template.html`) and run `python3 build_dashboard.py --inline-lib /path/to/chart.umd.js`.

`study.py` holds everything that depends on which study is being analysed (see below).
`common.py` is shared setup: paths, the colour palette, the tables it re-exports from
`study.py`, and `savefig()` (writes both PNG at 200 dpi and PDF).

## Two studies in one pipeline

The same scripts analyse the **monocular** study (D-NeRF, notebooks `01`-`03`) and the
**multi-view** one (Neural 3D Video, notebooks `05`-`07`). They are reported separately,
so everything that names a method or a scene — folders, labels, colours, scene lists, the
`FAIR` set, the `PARTIAL` and `NOT_EVAL` flags — and every sentence a figure prints about
the protocol (the caveat, how Protocol A is called, the GPU, the common budgets) lives in
`study.py` and is selected by `GS_STUDY`, while the data root is `GS_ROOT` as before:

```bash
./run_all.sh                                               # monocular (the default)
GS_ROOT="$(pwd)/../../n3dv" GS_STUDY=n3dv ./run_all.sh       # multi-view, 300 frames (main)
GS_ROOT="$(pwd)/../../n3dv" GS_STUDY=n3dv_f50 ./run_all.sh   # multi-view, 50 frames
```

Every output — tables, the 46 figures, `dashboard_data.json` and `benchmark_dashboard.html` —
is written under `$GS_ROOT/analysis/` (`$GS_ROOT/analysis_f50/` for `n3dv_f50`), so the
studies never overwrite each other. The two N3DV studies read the same method folders and
keep only the runs of their own window (`NUM_FRAMES` in `study.py`); the 300-frame one has
no Protocol B runs.

Three things are specific to the multi-view studies:

* **One folder, two methods.** Spacetime Gaussians is benchmarked in two variants whose
  runs share `spacetime_gaussians_output/`. `VARIANTS` in `study.py` names the config key
  that tells them apart (`stg_model`), and each variant becomes its own row and colour. A
  variant with no run is dropped from the figures (`VARIANT_GROUPS`).
* **No common step budget.** Protocol A runs each method at its official budget, so
  `STEP_BUDGET` is `None` and the "equal steps" row of figure 44 is left out; the titles
  say "official budget of each method" where the monocular ones say "equal-iteration".
* **The official readout.** `tables/table_official_readout.csv` reads each Protocol A run
  at the step its authors read it — recorded by the notebook as `official_test_iteration`
  (Spacetime Gaussians, per scene) or `official_iterations` — next to its end and its
  best, with the preparation time and the disk of the scene.

The grouped-bar figures size their bars from the number of methods, so four rows fit.
With `GS_STUDY` unset the pipeline behaves exactly as it did before `study.py` existed.

## Adding a new method or scene

1. Drop its output folder next to `4dgaussian_output/` etc. (under the data root of its
   study), with the same `<run>/benchmark/benchmark_*.json` layout.
2. `study.py`, in the branch of that study: add the folder to `METHOD_DIR` (folder name ->
   long label) and the long label to `SHORT` (-> short tag); add the short tag to `C`
   (colour), `ORDER` and `LABEL`; add new scenes to `SCENES` and `SCENE_LABEL`; mark
   incomplete runs in `PARTIAL` and non-evaluable method/scene pairs in `NOT_EVAL`; update
   `FAIR`, the scenes used for the aggregate means. Every figure honours all of it
   automatically, and the grouped bars are centred for any number of methods.
3. Re-run `./run_all.sh` (with the `GS_ROOT` / `GS_STUDY` of that study).

Colours come from a colourblind-safe categorical palette; the first three slots
(blue / orange / aqua) are validated for all-pairs separation and the fourth
(reddish purple) is the next slot of the same palette, used by the second Spacetime
Gaussians variant. Keep new methods on the
following slots rather than picking free-hand hues. The two methods that appear in
both studies keep their hue across them.

## Conventions baked into the numbers

* `iter_total` is `total_iterations`, so the 3 000 coarse steps of 4DGaussians are counted.
* Protocol B costs are read at the **first crossing** of the L1 target (`fc_*` columns in
  `runs_summary.csv`), not at run termination.
* `images_seen` = steps × batch size, the batch-size-corrected budget axis.
* Aggregates use the five scenes completed by all three methods.

## Overriding the data root

Scripts resolve the dataset root as two levels above this folder. To point them elsewhere:

```bash
GS_ROOT=/path/to/output_monocular_dataset ./run_all.sh
```
