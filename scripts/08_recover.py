"""Step 8. Fit every record that is not clean with TM and write the result table of each run.

Fitted: the pile-up classes (2nd trigger, slow rise, off diagonal) and records that start on the
tail of an earlier pulse (baseline). Clean pulses keep their template amplitude; saturated
pulses get a time from the fast-filter trigger and no amplitude. Every pulse gets its time since
the run start (record time from out.txt + position in the record) and its energy.

  runs/<run>/ch<c>/events.csv   one row per pulse (columns: README, section 4)

Figures: runs/.../figures/09_pileup_fits_random.png, 10_pileup_fits_crowded.png,
         11_energy_spectra.png, 12_time_distributions.png

    python scripts/08_recover.py [--runs ...] [--channels ...] [--limit N]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zgzpileup import io, plots, pulses, tm  # noqa: E402
from zgzpileup.project import banner, load_npz, need, parse, update  # noqa: E402

proj, args = parse(__doc__.split("\n")[0], lambda p: p.add_argument(
    "--limit", type=int, default=None, help="fit only this many records (quick test)"))
cfg = proj.cfg
C = pulses.CLASSES
KIND = {C.index("2nd trigger"): "fitted", C.index("slow rise"): "fitted", C.index("off diagonal"): "fitted",
        C.index("baseline"): "baseline"}
for run in proj.runs:
    for ch in proj.channels:
        banner(f"{run} channel {ch}: TM on the pile-up records")
        d = proj.run_dir(run, ch)
        md = proj.model_dir(proj.group_of(run), ch)
        need(d, "records.npz", "times.npz", "pulses.npz")
        need(md, "tm_model.npz")
        model_path = os.path.join(md, "tm_model.npz")
        rec = io.Records.load(os.path.join(d, "records.npz"))
        P = load_npz(os.path.join(d, "pulses.npz"))
        tz = np.load(os.path.join(d, "times.npz"))
        model = tm.Model.load(model_path)
        s = model.s
        cal_path = os.path.join(md, "calibration.json")
        cal = json.load(open(cal_path)) if os.path.exists(cal_path) else {}
        slope, offset = cal.get("slope"), cal.get("offset")
        cls, base, amp, t50 = P["cls"], P["baseline"].astype(float), P["amp"], P["t50"]
        dt, D, ref = rec.dt_us, s["D"], s["ref"]
        rtime, orow = tz["time"], tz["out_row"]

        todo = np.nonzero(np.isin(cls, list(KIND)))[0]
        if args.limit:
            todo = np.sort(np.random.default_rng(proj.seed).choice(todo, min(args.limit, len(todo)), replace=False))
        print(f"TM on {len(todo):,} records, {cfg.WORKERS} workers")
        t0 = time.time()
        waves = [rec.wave(i, base[i]) for i in todo]
        fits = tm.fit_many(waves, model_path, cfg.WORKERS, True, cfg.MAX_TEMP)
        fit_min = (time.time() - t0) / 60

        # one row per pulse
        rows = []
        for i in np.nonzero(cls == 0)[0]:
            rows.append((i, "clean", 1, 0, float(t50[i]), float(amp[i]), np.nan, None))
        for i, (nt, pk, chi2) in zip(todo, fits):
            pk = sorted(pk, key=lambda z: z[1])
            for j, (A, p) in enumerate(pk):
                flag = None
                if not 0 <= p < rec.length[i]:
                    flag = "outside_record"
                elif A >= 0.99 * s["max_A"] or A <= 0:
                    flag = "amplitude_at_bound"
                rows.append((i, KIND[cls[i]], len(pk), j, p - D, A, chi2, flag))
        for i in np.nonzero(cls == C.index("saturated"))[0]:
            tr = tm.triggers(rec.wave(i, base[i]), s["L"], s["ff_threshold"])
            if len(tr):
                rows.append((i, "saturated", np.nan, 0, float(tr[0] + s["peak2trig"] - D), np.nan, np.nan, "saturated"))
        ev = pd.DataFrame(rows, columns=["record", "kind", "n_pulses", "pulse", "t50_sample", "amplitude_adc", "chi2ndf", "flag"])
        ev["out_row"] = orow[ev.record.to_numpy()]
        ev["time_in_record_us"] = ev.t50_sample * dt
        ev["time_s"] = rtime[ev.record.to_numpy()] + (ev.t50_sample.to_numpy() - ref) * dt * 1e-6
        ev["energy_kev"] = slope * ev.amplitude_adc + offset if slope is not None else np.nan
        ev["relative_area"] = np.where(ev.kind == "clean", P["relative_area"][ev.record.to_numpy()], np.nan)
        ev = ev.sort_values(["record", "pulse"]).reset_index(drop=True)
        ev = ev[["record", "out_row", "kind", "n_pulses", "pulse", "time_s", "time_in_record_us", "amplitude_adc",
                 "energy_kev", "chi2ndf", "relative_area", "flag"]]
        ev.to_csv(os.path.join(d, "events.csv"), index=False, float_format="%.9g")

        # figures
        rng = np.random.default_rng(proj.seed)
        pu = [q for q, i in enumerate(todo) if KIND[cls[i]] == "fitted" and len(fits[q][1]) >= 2]
        pick = rng.choice(pu, min(16, len(pu)), replace=False)
        plots.fits_gallery(d, "09_pileup_fits_random.png", "TM fits of random pile-up records",
                           [waves[q] for q in pick], [fits[q] for q in pick], todo[pick], model, dt)
        crowd = sorted(pu, key=lambda q: -len(fits[q][1]))[:12]
        plots.fits_gallery(d, "10_pileup_fits_crowded.png", "TM fits of the most crowded pile-up records",
                           [waves[q] for q in crowd], [fits[q] for q in crowd], todo[crowd], model, dt, cols=3)
        plots.spectra(d, ev)
        plots.time_distributions(d, ev)
        good = ev.flag.isna()
        npr = ev[(ev.kind == "fitted")].groupby("record").size()
        multi = (ev.kind == "fitted") & (ev.n_pulses >= 2) & good
        res = dict(records_fitted=int(len(todo)), fit_minutes=round(fit_min, 1),
                   pulses={k_: int(((ev.kind == k_) & good).sum()) for k_ in ("clean", "fitted", "baseline")},
                   pulses_in_pileup_records=int(multi.sum()), pileup_records=int(ev[multi].record.nunique()),
                   saturated=int((ev.kind == "saturated").sum()),
                   flagged={str(f): int(n_) for f, n_ in ev.flag.value_counts().items()},
                   pulses_per_fitted_record={int(k_): int(v) for k_, v in npr.value_counts().sort_index().items()},
                   no_time=int(ev.time_s.isna().sum()))
        print(f"{res['pileup_records']:,} pile-up records resolved into {res['pulses_in_pileup_records']:,} pulses; "
              f"{len(ev):,} rows in events.csv; {fit_min:.1f} min")
        update(d, "recover", res)
