#!/usr/bin/env bash
# Regenerate every table, figure and the dashboard data from the raw benchmark JSON.
# Order matters: aggregate.py builds the CSVs that every other script reads.
set -e
cd "$(dirname "$0")"
python3 aggregate.py          # curves_all.csv + runs_summary.csv
python3 tables.py             # table_iso_iterations / iso_loss / aggregates_fair5
for f in fig_curves fig_compare fig_analysis fig_agg fig_extra fig_fair; do
  echo "== $f"; python3 "$f.py"
done
python3 export_web.py         # dashboard_data.json
python3 build_dashboard.py    # benchmark_dashboard.html (Chart.js from CDN)
echo "done — $(ls ../figures/*.png | wc -l) figures"
