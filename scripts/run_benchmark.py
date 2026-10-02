#!/usr/bin/env python3
"""Run Part 1 of a benchmark notebook from a terminal, with no browser.

The notebooks are written to be opened in a browser, but everything that produces a
measurement lives in **Part 1** (configuration, setup, dataset, training). Part 2 renders
and computes metrics again, and must never run while a training loop is in progress. This
script therefore executes the notebook **up to the Part 2 marker and no further**, with a
real Jupyter kernel, so that `!pip install`, `%cd` and every other notebook-ism behave
exactly as they do in the browser.

Every setting of cell 0.1 is read from a BENCH_<NAME> environment variable, so a run is
configured entirely from the command line:

    python3 scripts/run_benchmark.py notebooks/05_4dgaussians_wu_n3dv.ipynb \
        --set TRAINING_MODE=iterations --set SCENES=sear_steak,flame_steak

Output is streamed to the terminal as it is produced, and the executed notebook (with all
its output) is saved next to the log so a finished run can be inspected later.

Requires `nbclient` and `ipykernel`:  pip install nbclient ipykernel
"""

import argparse
import datetime
import os
import re
import sys

PART2_MARKERS = ("# Part 2: evaluation and visualisation",)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute Part 1 of a benchmark notebook headlessly.")
    parser.add_argument("notebook", help="path to the .ipynb file")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="set BENCH_<NAME>=<VALUE> for this run; repeatable")
    parser.add_argument("--log-dir", default="logs",
                        help="where to write the log and the executed notebook "
                             "(default: ./logs)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="seconds allowed per cell (default: no limit, which is what "
                             "a multi-hour training cell needs)")
    parser.add_argument("--kernel", default="python3", help="kernel name (default: python3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the settings and the cells that would run, then stop")
    parser.add_argument("--full", action="store_true",
                        help="run Part 2 as well; never use this during a training loop")
    return parser.parse_args()


def apply_settings(pairs):
    """Turn --set NAME=VALUE into BENCH_NAME=VALUE in the environment."""
    for pair in pairs:
        if "=" not in pair:
            sys.exit("--set expects NAME=VALUE, got %r" % pair)
        name, value = pair.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
            sys.exit("%r is not a valid setting name" % name)
        os.environ["BENCH_" + name.upper()] = value


def split_at_part2(notebook):
    """Return the cells up to the Part 2 marker, and how many were dropped."""
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            source = "".join(cell.source)
            if any(marker in source for marker in PART2_MARKERS):
                return notebook.cells[:index], len(notebook.cells) - index
    return notebook.cells, 0


def main():
    args = parse_args()
    apply_settings(args.set)

    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        sys.exit("nbclient and nbformat are required:\n"
                 "    pip install nbclient nbformat ipykernel")

    notebook = nbformat.read(args.notebook, as_version=4)
    total = len(notebook.cells)
    if not args.full:
        notebook.cells, dropped = split_at_part2(notebook)
    else:
        dropped = 0

    settings = sorted((k, v) for k, v in os.environ.items() if k.startswith("BENCH_"))
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.splitext(os.path.basename(args.notebook))[0]
    os.makedirs(args.log_dir, exist_ok=True)
    executed_path = os.path.join(args.log_dir, "%s.%s.executed.ipynb" % (base, stamp))

    print("=" * 78)
    print("notebook : %s" % args.notebook)
    print("cells    : %d of %d (Part 2 skipped: %d cells)"
          % (len(notebook.cells), total, dropped))
    print("settings : %s" % ("\n           ".join("%s=%s" % kv for kv in settings)
                             or "none (the notebook's own defaults)"))
    print("saving   : %s" % executed_path)
    print("=" * 78, flush=True)

    if args.dry_run:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                first = next((line for line in "".join(cell.source).split("\n")
                              if line.strip()), "")
                print("  [%2d] %s" % (index, first[:100]))
        return 0

    class StreamingClient(NotebookClient):
        """Print cell output while it is produced, instead of only at the end.

        nbclient collects output into the notebook and shows nothing until the run
        finishes, which is useless for a training cell that runs for hours. `output()`
        is called for every message the kernel emits, so echoing there gives a live log
        that `tail -f` can follow.
        """

        def output(self, outs, msg, display_id, cell_index):
            content = msg.get("content", {})
            msg_type = msg.get("msg_type")
            if msg_type == "stream":
                sys.stdout.write(content.get("text", ""))
            elif msg_type == "error":
                sys.stdout.write("\n".join(content.get("traceback", [])) + "\n")
            elif msg_type in ("execute_result", "display_data"):
                text = (content.get("data") or {}).get("text/plain")
                if text:
                    sys.stdout.write(text + "\n")
            sys.stdout.flush()
            return super().output(outs, msg, display_id, cell_index)

    client = StreamingClient(
        notebook,
        timeout=args.timeout,          # None = no per-cell limit
        kernel_name=args.kernel,
        allow_errors=False,
        resources={"metadata": {"path": os.path.dirname(os.path.abspath(args.notebook))}},
    )

    status = 0
    try:
        client.execute()
    except Exception as exc:
        print("\nFAILED: %r" % (exc,), file=sys.stderr)
        status = 1
    finally:
        import nbformat as _nbformat
        _nbformat.write(notebook, executed_path)
        print("\nExecuted notebook saved to %s" % executed_path)

    return status


if __name__ == "__main__":
    sys.exit(main())
