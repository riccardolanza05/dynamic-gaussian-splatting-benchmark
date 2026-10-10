# Multi-view results (Neural 3D Video)

This folder is the data root of the **multi-view** study: three dynamic Gaussian Splatting
methods on the six scenes of *Neural 3D Video Synthesis from Multi-View Video*, `cam00`
held out, 1352×1014, on all 300 frames (the main study) and on the first 50 (the secondary
one). The protocol is documented in
[`docs/METHODOLOGY_MULTIVIEW.md`](../../docs/METHODOLOGY_MULTIVIEW.md).

**It is empty: no multi-view training run has been made yet.** The notebooks
[`05`](../../notebooks/05_4dgaussians_wu_n3dv.ipynb)–[`07`](../../notebooks/07_spacetime_gaussians_li_n3dv.ipynb)
write here, one folder per method:

```
results/n3dv/
├── 4dgaussian_n3dv_output/       notebook 05
├── 4dgs_fudan_n3dv_output/       notebook 06
├── spacetime_gaussians_output/   notebook 07, both variants (lite and full)
├── analysis/                     created by the pipeline: main study, 300 frames
└── analysis_f50/                 created by the pipeline: secondary study, 50 frames
```

with the same `<run>/benchmark/benchmark_<protocol>.json` layout as the monocular study.

**How the folder gets filled.** The notebooks do not write here directly: they write to the
machine that has the GPU, and push each finished run to **one shared Google Drive folder**
(`dgs-benchmark-n3dv` by default) that gathers the JSON of all the methods in exactly this
layout. Bringing it here is one command on your own machine:

```bash
rclone copy gdrive:dgs-benchmark-n3dv results/n3dv --progress
```

How the runs are launched is summarised in the
[running notes of the main README](../../README.md#running-on-a-cloud-gpu-over-ssh).

**Run folders say what they are.** The name carries the frame window and the budget —
`<scene>_f300_iters14000` for a Protocol A run of 4DGaussians on the main window,
`<scene>_f50_loss<target>` for Protocol B — so runs made with different windows cannot be
mixed. The two Spacetime Gaussians variants share one method folder: `ours_full` runs end in
`_full` (`<scene>_f300_iters30000_full`), `ours_lite` runs have no tag, and every JSON
records its variant (`stg_model`). A Spacetime Gaussians run at 300 frames also has a
`blocks/` folder with the JSON of each of its six 50-frame models; the analysis reads only
the merged `benchmark/` JSON.

To build the tables, figures and dashboard once the runs are here:

```bash
cd results/analysis/scripts
GS_ROOT="$(pwd)/../../n3dv" GS_STUDY=n3dv ./run_all.sh        # main study, 300 frames
GS_ROOT="$(pwd)/../../n3dv" GS_STUDY=n3dv_f50 ./run_all.sh    # secondary study, 50 frames
```

Everything lands in `results/n3dv/analysis/` and `results/n3dv/analysis_f50/`, leaving the
monocular study untouched. Runs of both windows sit side by side in the same method folders;
each study keeps only the runs of its own window. The two Spacetime Gaussians variants are
reported as two methods, and a variant with no run is left out. Besides the tables of the
monocular study, the pipeline writes `tables/table_official_readout.csv`: each Protocol A
run read at the step its authors read it (the per-scene test snapshot for Spacetime
Gaussians), next to its end and its best. The folder-to-label mapping is in
`results/analysis/scripts/study.py`.
