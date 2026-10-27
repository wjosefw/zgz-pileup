"""Step 1. Read every run and channel into a compact file and give every record its time.

  c<ch>.zip / .txt  ->  runs/<run>/ch<c>/records.npz   the samples of every record
  out.txt           ->  runs/<run>/ch<c>/times.npz     time of every record since the run start

The records in c<ch>.txt carry no time; out.txt has one row per record (channel, time, VST
energy). Records and rows are matched by the VST energy, which is the maximum of the third
column of c<ch>.txt (io.align_times). Figure: 01_record_gaps.png (time between records).

    python scripts/01_read_data.py [--runs RUN1 ...] [--channels 0 1 ...]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zgzpileup import io, plots  # noqa: E402
from zgzpileup.project import banner, parse, update  # noqa: E402

proj, _ = parse(__doc__.split("\n")[0])
for run, folder in proj.runs.items():
    out_txt = os.path.join(folder, "out.txt")
    for ch in proj.channels:
        banner(f"{run} channel {ch}")
        d = proj.run_dir(run, ch)
        t0 = time.time()
        src = io.find_file(folder, ch, ["c{ch}.zip", "c{ch}.txt"])
        if src is None:
            raise FileNotFoundError(f"no c{ch}.zip or c{ch}.txt in {folder}")
        print(f"reading {src}")
        rec = io.read_channel(src)
        rec.save(os.path.join(d, "records.npz"))
        update(d, "extract", dict(source=src, records=len(rec), samples=int(rec.adc.size), dt_us=rec.dt_us,
                                  adc_step=rec.adc_step, adc_max=int(rec.adc.max()),
                                  length_min_median_max=[int(rec.length.min()), int(np.median(rec.length)),
                                                         int(rec.length.max())]))
        # times from out.txt
        o = io.read_out_txt(out_txt, ch)
        row, st = io.align_times(rec.fmax, o.E.to_numpy())
        t = np.full(len(rec), np.nan)
        ok = row >= 0
        t[ok] = o.t.to_numpy()[row[ok]]
        np.savez(os.path.join(d, "times.npz"), time=t, out_row=row)
        gap = np.diff(t[ok]) * 1e6
        plots.gaps(d, gap)
        update(d, "times", dict(out_txt=out_txt, out_rows=len(o), matched=int(ok.sum()), matched_fraction=float(ok.mean()),
                                **st, monotonic=bool(np.all(np.diff(t[ok]) >= 0)),
                                run_length_h=float((t[ok][-1] - t[ok][0]) / 3600),
                                fraction_of_gaps_below_400us=float(np.mean(gap < 400))))
        print(f"{len(rec):,} records; {ok.sum():,} matched to out.txt ({ok.mean():.2%}); "
              f"{(time.time() - t0) / 60:.1f} min")
