"""Step 3. Measure the pulse tail and undershoot, which the records cut off.

Records stop while the pulse is still in its tail (~4 ADC above baseline), so the tail and the
undershoot after it are not in any record. When a record starts shortly after an isolated clean
pulse, its pre-trigger baseline is that pulse's tail at that moment: baseline / amplitude against
the time since the pulse (from out.txt) traces the tail out to several hundred µs (zgzpileup/tail.py).
Record pairs from all runs of the group are pooled.

  models/<group>/ch<c>/tail.npz   tail curve per unit amplitude, in samples after the reference point

Figure: models/.../figures/05_tail_undershoot.png (redrawn with the full template by step 5)

    python scripts/03_tail.py [--channels ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zgzpileup import plots, pulses, tail as tailmod  # noqa: E402
from zgzpileup.project import banner, load_npz, need, parse, update  # noqa: E402

proj, _ = parse(__doc__.split("\n")[0])
for ch in proj.channels:
    for group, runs in proj.groups().items():
        banner(f"channel {ch}: tail from {', '.join(runs)}")
        md = proj.model_dir(group, ch)
        need(md, "shape.npz")
        S = np.load(os.path.join(md, "shape.npz"))
        shape, ref, dt = S["shape"], int(S["ref"]), float(S["dt_us"])
        D = pulses.shape_times(shape, ref)["D"]
        lengths = {}
        for r in runs:
            d = proj.run_dir(r, ch)
            need(d, "records.npz", "times.npz", "pulses.npz")
            lengths[r] = np.load(os.path.join(d, "records.npz"))["length"]
        K1 = max(int(v.max()) for v in lengths.values()) + 50     # template reaches past the longest record
        max_gap = max(400.0, 1.3 * K1 * dt)
        ks, ys, zeros = [], [], []
        for r in runs:
            d = proj.run_dir(r, ch)
            P = load_npz(os.path.join(d, "pulses.npz"))
            t = np.load(os.path.join(d, "times.npz"))["time"]
            k, y, zero = tailmod.baseline_pairs(t, lengths[r], P["cls"], P["baseline"].astype(float), P["amp"], P["t50"],
                                                D, dt, max_gap)
            print(f"{r}: {len(k):,} record pairs, baseline zero point {zero:.2f} ADC")
            ks.append(k); ys.append(y); zeros.append(zero)
        k, y = np.concatenate(ks), np.concatenate(ys)
        if len(k) < 2000:
            raise RuntimeError(f"only {len(k)} record pairs for the tail measurement")
        cen, smooth, med, err, cnt = tailmod.tail_curve(k, y)
        K0 = int(np.ceil(cen[0] + 2 * tailmod.BIN))
        if cen[-1] < K1:
            print(f"warning: tail curve ends at {cen[-1] * dt:.0f} µs, before the longest record "
                  f"({K1 * dt:.0f} µs): extended with the exponential decay of its last 50 µs")
            m = cen > cen[-1] - 50 / dt
            tau = -1 / np.polyfit(cen[m], np.log(np.abs(smooth[m]) + 1e-6), 1)[0]
            ext = np.arange(cen[-1] + tailmod.BIN, K1 + tailmod.BIN, tailmod.BIN)
            cen = np.r_[cen, ext]; smooth = np.r_[smooth, smooth[-1] * np.exp(-(ext - cen[-len(ext) - 1]) / tau)]
        np.savez(os.path.join(md, "tail.npz"), k=k, y=y, cen=cen, smooth=smooth, med=med, err=err, cnt=cnt,
                 zero=np.mean(zeros), K0=K0, K1=K1)
        fin = np.isfinite(shape)
        plots.tail(md, k, y, cen, smooth, med, err, (np.arange(len(shape)) - ref - D)[fin], shape[fin], None, None, K0, dt)
        neg = smooth < 0
        zc = float(cen[np.argmax(neg & (cen > 0))] * dt) if neg.any() else None
        i = int(np.argmin(smooth))
        res = dict(pairs=int(len(k)), zero_point_adc=float(np.mean(zeros)), join_us=K0 * dt, zero_crossing_us=zc,
                   minimum=float(smooth[i]), minimum_at_us=float(cen[i] * dt),
                   value_at_300us=float(np.interp(300 / dt, cen, smooth)), curve_ends_us=float(cen[-1] * dt))
        print(f"tail: zero crossing {'none' if zc is None else f'{zc:.1f}'} µs, minimum {smooth[i]:.3f} of the amplitude at {cen[i] * dt:.0f} µs")
        update(md, "tail", res)
