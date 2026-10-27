"""What every script needs: the settings of config.py, the runs and channels to work on, where
files go, and the summary.json files that collect the numbers of each step.

Folders inside WORK:
  runs/<run>/ch<c>/      one run of one channel: records, times, classes, events.csv, figures
  models/<group>/ch<c>/  what is learned from the clean pulses of a channel: pulse shape, tail,
                         TM settings and templates, calibration, benchmark, figures

A group is the set of runs whose clean pulses are pooled to learn one model per channel: all the
runs (POOL_RUNS = True in config.py, group name 'all') or each run on its own (group = run name).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    path = path or os.path.join(ROOT, "config.py")
    spec = importlib.util.spec_from_file_location("zgz_config", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


class Project:
    def __init__(self, cfg, channels=None, runs=None):
        self.cfg = cfg
        self.channels = list(channels) if channels else list(cfg.CHANNELS)
        self.all_runs = dict(cfg.RUNS)
        self.runs = {r: self.all_runs[r] for r in runs} if runs else dict(self.all_runs)
        self.work = cfg.WORK
        self.seed = getattr(cfg, "SEED", 0)

    def groups(self):
        """{group: [runs]} for the selected runs."""
        if getattr(self.cfg, "POOL_RUNS", True):
            return {"all": list(self.all_runs)}
        return {r: [r] for r in self.runs}

    def group_of(self, run):
        return "all" if getattr(self.cfg, "POOL_RUNS", True) else run

    def run_dir(self, run, ch):
        d = os.path.join(self.work, "runs", run, f"ch{ch}")
        os.makedirs(d, exist_ok=True)
        return d

    def model_dir(self, group, ch):
        d = os.path.join(self.work, "models", group, f"ch{ch}")
        os.makedirs(d, exist_ok=True)
        return d


def parse(description, extra=None):
    """Command line common to all scripts; returns (project, args)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="settings file (default: config.py next to the scripts folder)")
    p.add_argument("--channels", type=int, nargs="+", default=None, help="only these channels (default: CHANNELS)")
    p.add_argument("--runs", nargs="+", default=None, help="only these runs (default: all RUNS)")
    if extra:
        extra(p)
    a = p.parse_args()
    return Project(load_config(a.config), a.channels, a.runs), a


# ------------------------------------------------------------------ files
def need(d, *names):
    for n in names:
        if not os.path.exists(os.path.join(d, n)):
            raise FileNotFoundError(f"{os.path.join(d, n)} not found: run the script that makes it first")


def load_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def summary(d):
    p = os.path.join(d, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def update(d, key, value):
    s = summary(d)
    s[key] = value
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(s, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def banner(text):
    print(f"\n=== {text} ===", flush=True)
