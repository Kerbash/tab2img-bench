"""Command line:

    python -m tab2img methods                          list available T2I methods
    python -m tab2img run configs/example.yaml         run the benchmark in a config
    python -m tab2img characteristics configs/example.yaml -o chars.csv
    python -m tab2img analyze results/ [more dirs] --report report.txt --cd cd.png

Config format (YAML or JSON); dataset paths are relative to the config file:

    output_dir: results
    protocol: {folds: 10, seed: 42, epochs: 30, cnn: simple, clip: false}
    methods: [SOM2IMG, CorrelationBased, {method: SOM2IMG, params: {grid_mult: 2}, label: SOM_x2}]
    baselines: [LogReg_L2, RandomForest]
    datasets:
      - {name: BreastCancer, csv: data/bc.csv, target: Diagnosis, exclude: [NCTD]}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_config(path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    cfg["_base"] = path.resolve().parent
    return cfg


def _dataset_loaders(cfg):
    from tab2img.data import load_csv
    base = cfg["_base"]
    out = []
    for d in cfg["datasets"]:
        csv = Path(d["csv"])
        csv = csv if csv.is_absolute() else base / csv
        out.append(lambda csv=csv, d=d: load_csv(csv, d.get("target", "target"), d.get("name")))
    return out


def cmd_run(args):
    from tab2img.evaluation import Protocol, run_benchmark
    from tab2img.methods import PAPER_METHODS
    from tab2img.models.tabular import BASELINES
    cfg = load_config(args.config)
    out = Path(cfg.get("output_dir", "results"))
    out = out if out.is_absolute() else cfg["_base"] / out
    exclude = {d.get("name") or Path(d["csv"]).stem: d.get("exclude", []) for d in cfg["datasets"]}
    run_benchmark(_dataset_loaders(cfg), out,
                  methods=cfg.get("methods", PAPER_METHODS),
                  baselines=cfg.get("baselines", BASELINES),
                  protocol=Protocol(**cfg.get("protocol", {})),
                  exclude=exclude, overwrite=args.overwrite)
    print(f"done -> {out}")


def cmd_characteristics(args):
    from tab2img.analysis import characteristics_table
    t = characteristics_table(_dataset_loaders(load_config(args.config)))
    t.to_csv(args.output)
    print(t.round(3).to_string())


def cmd_analyze(args):
    import pandas as pd
    from tab2img.analysis import full_report, load_results
    from tab2img.analysis.plots import cd_diagram
    rs = load_results(*args.dirs, metric=args.metric)
    ch = pd.read_csv(args.characteristics).set_index("dataset") if args.characteristics else None
    print(full_report(rs, focus=args.focus, characteristics=ch, cutoff=args.cutoff, path=args.report))
    if args.cd:
        cd_diagram(rs.select(methods=rs.t2i_methods()).scores, args.cd)
        print(f"CD diagram -> {args.cd}")


def cmd_methods(args):
    from tab2img.methods import PAPER_METHODS, available
    for m in available():
        print(f"{m:18s}{'  (paper)' if m in PAPER_METHODS else ''}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="tab2img", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run", help="run a benchmark config")
    s.add_argument("config")
    s.add_argument("--overwrite", action="store_true", help="re-run datasets that already have results")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("characteristics", help="compute dataset characteristics for a config")
    s.add_argument("config")
    s.add_argument("-o", "--output", default="dataset_characteristics.csv")
    s.set_defaults(fn=cmd_characteristics)
    s = sub.add_parser("analyze", help="statistical report over result folders")
    s.add_argument("dirs", nargs="+")
    s.add_argument("--focus", default="SOM2IMG")
    s.add_argument("--metric", default="f1", choices=["f1", "acc", "auc"])
    s.add_argument("--cutoff", type=float, default=1.0, help="d/N regime split")
    s.add_argument("--characteristics", help="CSV with a dataset column and dN (optional)")
    s.add_argument("--report", help="write the report to this file")
    s.add_argument("--cd", help="write a critical-difference diagram PNG")
    s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("methods", help="list T2I methods")
    s.set_defaults(fn=cmd_methods)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
