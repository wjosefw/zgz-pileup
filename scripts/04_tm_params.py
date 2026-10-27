"""Step 4. Choose the TM trigger settings from the clean pulses.

The fast filter (mean of the next L samples minus mean of the previous L) finds the pulses that
TM then fits. L is the length with the best signal-to-noise ratio that still fits in the
pre-trigger samples before the pulse; the threshold is the lowest at which no more than 0.2 % of
clean pulses give more than one trigger (tm.characterise). Measured on 20,000 training pulses
(clean, not held out), pooled over the runs of the group.

  models/<group>/ch<c>/tm_params.json   L, threshold, peak-to-trigger distance, and the scans

Figure: models/.../figures/06_tm_settings.png (the templates panel is added by step 5)

    python scripts/04_tm_params.py [--channels ...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zgzpileup import io, plots, pulses, tm  # noqa: E402
from zgzpileup.project import banner, load_npz, need, parse, update  # noqa: E402

proj, _ = parse(__doc__.split("\n")[0])
for ch in proj.channels:
    for group, runs in proj.groups().items():
        banner(f"channel {ch}: TM trigger settings")
        md = proj.model_dir(group, ch)
        need(md, "shape.npz")
        S = np.load(os.path.join(md, "shape.npz"))
        shape, ref, dt = S["shape"], int(S["ref"]), float(S["dt_us"])
        st = pulses.shape_times(shape, ref)
        P = {r: load_npz(os.path.join(proj.run_dir(r, ch), "pulses.npz")) for r in runs}
        rng = np.random.default_rng(proj.seed)
        _, sub = tm.training_selection({r: np.nonzero((P[r]["cls"] == 0) & ~P[r]["holdout"])[0] for r in runs}, rng)
        # the pulses in the (random) order of `sub`, reading one run at a time
        waves, A, t50 = [None] * len(sub), np.zeros(len(sub)), np.zeros(len(sub), int)
        for r in runs:
            rec = io.Records.load(os.path.join(proj.run_dir(r, ch), "records.npz"))
            for j, (r_, i) in enumerate(sub):
                if r_ == r:
                    waves[j], A[j], t50[j] = rec.wave(i, P[r]["baseline"][i]), P[r]["amp"][i], P[r]["t50"][i]
            del rec
        s, scans = tm.characterise(waves, A, t50, shape, ref, st["D"], st["t10_minus_t50"], rng)
        json.dump(dict(settings=s, scans=scans), open(os.path.join(md, "tm_params.json"), "w"), indent=1)
        plots.tm_settings(md, scans, s, None, dt)
        print(f"fast filter L = {s['L']} samples, threshold {s['ff_threshold']:.2f} ADC ({s['threshold_sigma']:.1f} σ), "
              f"peak-to-trigger {s['peak2trig']} samples")
        update(md, "tm_params", s)
