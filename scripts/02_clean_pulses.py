"""Step 2. Sort every record into a class, select the clean single pulses and learn the pulse shape.

Every record of every run is put in one class (first rule that applies): saturated, baseline
(starts on the tail of an earlier pulse), no pulse, 2nd trigger, slow rise, off diagonal (area
above the single-pulse band) or clean (see zgzpileup/pulses.py). The clean pulses of a channel,
pooled over the runs of its group, give the median pulse shape, and every clean pulse gets its
template amplitude (least-squares scale of that shape). The last 10 % of the clean pulses of
each run are held out of the templates for the benchmark.

  runs/<run>/ch<c>/pulses.npz    classes, baselines, t50, amplitudes, hold-out flag
  models/<group>/ch<c>/shape.npz pulse shape (t50 at sample `ref`), saturation level, noise

When runs are pooled, the photopeak position of every run is printed: if it moves by more than
~1 %, the gain changed between runs and they should not be pooled (POOL_RUNS = False).

Figures: runs/.../figures/02_clean_selection.png, 03_record_classes.png;
         models/.../figures/04_clean_pulses.png

    python scripts/02_clean_pulses.py [--runs ...] [--channels ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zgzpileup import io, plots, pulses  # noqa: E402
from zgzpileup.project import banner, need, parse, update  # noqa: E402

HOLDOUT = 0.1          # last fraction of the clean pulses of each run, kept for the benchmark
N_SHAPE = 200_000      # clean pulses used for the median shape

proj, _ = parse(__doc__.split("\n")[0])
for ch in proj.channels:
    for group, runs in proj.groups().items():
        md = proj.model_dir(group, ch)
        cache = {}

        def records(run):
            if run in cache:
                return cache[run]
            d = proj.run_dir(run, ch)
            need(d, "records.npz")
            rec = io.Records.load(os.path.join(d, "records.npz"))
            if len(runs) == 1:
                cache[run] = rec      # one run: keep it in memory between passes
            return rec

        # --- classes, run by run
        R = {}
        for run in runs:
            banner(f"{run} channel {ch}: classes")
            d = proj.run_dir(run, ch)
            rec = records(run)
            sat = int(rec.adc.max())
            F = pulses.record_features(rec)
            cls, band = pulses.classify(F, sat)
            idx = np.nonzero(cls == 0)[0]
            rng = np.random.default_rng(proj.seed)
            plots.selection(d, rec, F, cls, band, pulses.CLASSES, pulses.BAND_K, rec.dt_us, rng)
            plots.classes_gallery(d, rec, cls, F["base"], pulses.CLASSES, rec.dt_us, rng)
            counts = {pulses.CLASSES[c]: int(np.sum(cls == c)) for c in range(len(pulses.CLASSES))}
            print(f"classes: {counts}")
            update(d, "clean", dict(classes=counts, saturation=sat,
                                    selection_trigger=dict(L=pulses.SEL_L, noise_adc=F["sel_noise"],
                                                           threshold_adc=F["sel_threshold"], sigma=pulses.SEL_SIGMA),
                                    band=dict(coefficients=[float(c) for c in band["coef"]], k=pulses.BAND_K),
                                    second_shape_fraction_of_clean=float(np.mean(band["relative_area"][idx] < 0.92)),
                                    clean_length_min_median_max=[int(rec.length[idx].min()), int(np.median(rec.length[idx])),
                                                                 int(rec.length[idx].max())]))
            noise = [np.std(rec.adc[rec.offset[i]:rec.offset[i] + pulses.N_BASELINE])
                     for i in idx[:: max(1, len(idx) // 20000)]]
            R[run] = dict(cls=cls, base=F["base"], amp_s=F["amp"], t50=F["t50"], idx=idx, sat=sat, noise=noise,
                          lmax=int(rec.length[idx].max()), dt=rec.dt_us, F=F, band=band)
            del rec

        # --- pulse shape from the pooled clean pulses: first normalised by the smoothed maximum,
        # then by the template amplitudes of the first pass (less noisy)
        banner(f"channel {ch}: pulse shape from {', '.join(runs)}")
        sats = {r: R[r]["sat"] for r in runs}
        if len(set(sats.values())) > 1:
            print(f"warning: different saturation levels between runs: {sats}")
        sat = max(sats.values())
        ref = int(np.median(np.concatenate([R[r]["t50"][R[r]["idx"]] for r in runs])))
        Ls = max(R[r]["lmax"] for r in runs) + ref
        n_clean = sum(len(R[r]["idx"]) for r in runs)

        def pooled_shape(amp_key):
            rng = np.random.default_rng(proj.seed)
            stacks = []
            for r in runs:
                idx = R[r]["idx"]
                n = min(len(idx), int(round(N_SHAPE * len(idx) / n_clean)))
                sel = rng.choice(idx, n, replace=False)
                stacks.append(pulses.aligned_stack(records(r), sel, R[r]["base"], R[r][amp_key], R[r]["t50"], ref, Ls))
            return pulses.shape_from_stack(np.concatenate(stacks))

        shape, count = pooled_shape("amp_s")
        for r in runs:
            a = np.full(len(R[r]["cls"]), np.nan, np.float32)
            a[R[r]["idx"]] = pulses.template_amplitudes(records(r), R[r]["idx"], R[r]["base"], R[r]["t50"], shape, ref)
            R[r]["amp"] = a
        shape, count = pooled_shape("amp")
        peaks = {}
        for r in runs:
            idx = R[r]["idx"]
            R[r]["amp"][idx] = pulses.template_amplitudes(records(r), idx, R[r]["base"], R[r]["t50"], shape, ref)
            peaks[r] = pulses.photopeak(R[r]["amp"][idx])[0]
        st = pulses.shape_times(shape, ref)
        noise = float(np.median(np.concatenate([R[r]["noise"] for r in runs])))

        # --- save, run by run (classes and amplitudes) and for the group (shape)
        for r in runs:
            d = proj.run_dir(r, ch)
            idx, F, band = R[r]["idx"], R[r]["F"], R[r]["band"]
            holdout = np.zeros(len(R[r]["cls"]), bool)
            holdout[idx[int(len(idx) * (1 - HOLDOUT)):]] = True
            np.savez(os.path.join(d, "pulses.npz"), cls=R[r]["cls"], baseline=R[r]["base"], amp_smooth=R[r]["amp_s"],
                     t50=R[r]["t50"], amp=R[r]["amp"], holdout=holdout, area=F["area"], n_triggers=F["ntr"],
                     rise=F["rise"], band_coef=band["coef"], band_amp=band["amp"], band_spread=band["spread"],
                     relative_area=band["relative_area"])
        np.savez(os.path.join(md, "shape.npz"), shape=shape, shape_count=count, ref=ref, saturation=sat, noise=noise,
                 dt_us=R[runs[0]]["dt"])
        spread = 100 * (max(peaks.values()) / min(peaks.values()) - 1)
        print("photopeak (template amplitude, ADC) per run: " + ", ".join(f"{r} {v:.2f}" for r, v in peaks.items()))
        if len(runs) > 1 and spread > 1.0:
            print(f"WARNING: the photopeak moves by {spread:.1f} % between runs; the gain is not the same, "
                  f"set POOL_RUNS = False in config.py")
        r0 = runs[0]
        rec = records(r0)
        rng = np.random.default_rng(proj.seed)
        sel = np.sort(rng.choice(R[r0]["idx"], min(100_000, len(R[r0]["idx"])), replace=False))
        batch = pulses.padded_batch(rec, sel, R[r0]["base"], R[r0]["amp"][sel], R[r0]["t50"], shape, ref, R[r0]["lmax"])
        plots.clean_overview(md, batch, R[r0]["amp"][sel], shape, ref, rec.length[R[r0]["idx"]], R[r0]["amp"][R[r0]["idx"]],
                             rec.fmax, rec.fmax[R[r0]["idx"]], rec.dt_us)
        amp_all = np.concatenate([R[r]["amp"][R[r]["idx"]] for r in runs])
        res = dict(runs=runs, clean_pulses={r: int(len(R[r]["idx"])) for r in runs}, saturation=sat, t50_median=ref,
                   baseline_noise_adc=noise, amplitude_median=float(np.median(amp_all)), photopeak_adc_per_run=peaks,
                   photopeak_spread_pct=spread, **st)
        print(f"shape: t50 at sample {ref}; {st}; baseline noise {noise:.2f} ADC")
        update(md, "shape", res)
