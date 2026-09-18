#!/usr/bin/env python3
"""Assemble the interactive dashboard from dash_template.html + dashboard_data.json.

    python3 build_dashboard.py                     -> benchmark_dashboard.html, Chart.js from the CDN
    python3 build_dashboard.py --inline-lib X.js   -> same page with Chart.js embedded (works offline)

Run export_web.py first so dashboard_data.json is up to date.
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.dirname(HERE)                       # .../analysis
CDN  = '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>'

ap = argparse.ArgumentParser()
ap.add_argument("--inline-lib", help="path to a local chart.umd.js to embed instead of the CDN tag")
ap.add_argument("--out", default=os.path.join(OUT, "benchmark_dashboard.html"))
a = ap.parse_args()

tpl  = open(os.path.join(HERE, "dash_template.html")).read()
data = open(os.path.join(OUT, "dashboard_data.json")).read()
body = tpl.replace("__DATA__", data)

if a.inline_lib:
    body = body.replace(CDN, "<script>" + open(a.inline_lib).read() + "</script>")

he = body.index("</style>") + len("</style>")
html = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        + body[:he] + '</head><body style="margin:0">' + body[he:] + '</body></html>')
open(a.out, "w").write(html)
print("wrote", a.out, f"({len(html)/1024:.0f} KB)")
