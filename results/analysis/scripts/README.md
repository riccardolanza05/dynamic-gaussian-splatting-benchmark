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

`common.py` is shared setup: paths, the colour palette, the method/scene ordering, the
`partial` and `not evaluable` flags, and `savefig()` (writes both PNG at 200 dpi and PDF).

## Adding a new method or scene

1. Drop its output folder next to `4dgaussian_output/` etc., with the same
   `<run>/benchmark/benchmark_*.json` layout.
2. `aggregate.py` — add the folder to `METHOD_DIR` (folder name -> long label) and
   `SHORT` (long label -> short tag).
3. `common.py` — add the short tag to `C` (colour), `ORDER`, `LABEL`; add new scenes to
   `SCENES` and `SCENE_LABEL`. Mark incomplete runs in `PARTIAL` and non-evaluable
   method/scene pairs in `NOT_EVAL`; every figure honours both automatically.
4. `fig_agg.py`, `fig_analysis.py`, `fig_fair.py` — `FAIR` lists the scenes used for the
   aggregate means (those completed by every method). Update it.
5. Re-run `./run_all.sh`.

Colours come from a colourblind-safe categorical palette; the first three slots
(blue / orange / aqua) are validated for all-pairs separation. Keep new methods on the
next slots of that palette rather than picking free-hand hues.

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
