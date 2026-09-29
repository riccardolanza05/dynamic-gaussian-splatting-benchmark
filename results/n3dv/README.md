# Multi-view results (Neural 3D Video)

This folder is the data root of the **multi-view** study: four dynamic Gaussian Splatting
methods on the six scenes of *Neural 3D Video Synthesis from Multi-View Video*, first 50
frames, `cam00` held out, 1352×1014. The protocol is documented in
[`docs/METHODOLOGY_MULTIVIEW.md`](../../docs/METHODOLOGY_MULTIVIEW.md).

**It is empty: no multi-view training run has been made yet.** The notebooks
[`04`](../../notebooks/04_dynamic3dgaussians_luiten_n3dv.ipynb)–[`07`](../../notebooks/07_spacetime_gaussians_li_n3dv.ipynb)
write here, one folder per method:

```
results/n3dv/
├── dynamic3dgaussians_output/    notebook 04
├── 4dgaussian_n3dv_output/       notebook 05
├── 4dgs_fudan_n3dv_output/       notebook 06
├── spacetime_gaussians_output/   notebook 07
└── analysis/                     created by the pipeline
```

with the same `<run>/benchmark/benchmark_<protocol>.json` layout as the monocular study.

**How the folder gets filled.** The notebooks do not write here directly: they write to the
machine that has the GPU, and push each finished run to **one shared Google Drive folder**
(`dgs-benchmark-n3dv` by default) that gathers the JSON of all four methods in exactly this
layout. Bringing it here is one command on your own machine:

```bash
rclone copy gdrive:dgs-benchmark-n3dv results/n3dv --progress
```

How the runs are launched is summarised in the
[running notes of the main README](../../README.md#running-on-a-cloud-gpu-over-ssh).
Run folders carry the frame window in their name — `<scene>_f50_iters30000` for Protocol A,
`<scene>_f50_loss<target>` for Protocol B — so runs made with different windows cannot be
mixed.

To build the tables, figures and dashboard once the runs are here:

```bash
cd results/analysis/scripts
GS_ROOT="$(pwd)/../../n3dv" GS_STUDY=n3dv ./run_all.sh        # 50 frames, four methods
GS_ROOT="$(pwd)/../../n3dv" GS_STUDY=n3dv_full ./run_all.sh   # 300 frames, three methods
```

Everything lands in `results/n3dv/analysis/` (and `results/n3dv/analysis_f300/` for the
full-length study), leaving the monocular study untouched. Runs of both windows sit side by
side in the same method folders; each study keeps only the runs of its own window.
A Spacetime Gaussians run at 300 frames also has a `blocks/` folder with the JSON of each of
its six 50-frame models; the analysis reads only the merged `benchmark/` JSON.
The folder-to-label mapping is in `results/analysis/scripts/study.py`, in its `n3dv` branch.
