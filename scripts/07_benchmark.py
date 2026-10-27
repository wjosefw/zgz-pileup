"""Step 7. Benchmark TM on synthetic pile-up with known truth.

Pairs of held-out clean pulses (never used for the templates) are added with a delay drawn
log-uniformly between 0.2 and 100 µs, each continued after its record end with its amplitude
times the template tail plus noise, and fitted like real records. A pulse counts as found if a
fitted pulse lies within 4 samples of it with an amplitude within 10 %. Single held-out pulses
are also fitted, to measure the amplitude and time resolution without pile-up.

  models/<group>/ch<c>/benchmark.csv   efficiency and errors per delay bin

Figure: models/.../figures/08_benchmark.png

    python scripts/07_benchmark.py [--channels ...]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zgzpileup import io, plots, tm  # noqa: E402
from zgzpileup.project import banner, load_npz, need, parse, summary, update  # noqa: E402

TOL = 4                # samples
AMP_TOL = 0.10

proj, _ = parse(__doc__.split("\n")[0])
cfg = proj.cfg
for ch in proj.channels:
    for group, runs in proj.groups().items():
        banner(f"channel {ch}: benchmark")
        md = proj.model_dir(group, ch)
        need(md, "tm_model.npz")
        model_path = os.path.join(md, "tm_model.npz")
        model = tm.Model.load(model_path)
        s = model.s
        D, dt = s["D"], s["dt_us"]
        noise = summary(md).get("shape", {}).get("baseline_noise_adc", 0.6)
        P = {r: load_npz(os.path.join(proj.run_dir(r, ch), "pulses.npz")) for r in runs}
        ho = {r: np.nonzero(P[r]["holdout"])[0] for r in runs}
        n_ho = sum(len(v) for v in ho.values())
        rng = np.random.default_rng(proj.seed + 1)
        waves, truth, delays, swaves, s_truth = [], [], [], [], []
        for r in runs:                        # pairs are made within a run, in proportion to its hold-out pulses
            rec = io.Records.load(os.path.join(proj.run_dir(r, ch), "records.npz"))
            base, amp, t50 = P[r]["baseline"].astype(float), P[r]["amp"], P[r]["t50"]
            n = int(round(cfg.N_BENCHMARK * len(ho[r]) / n_ho))
            a, b = rng.choice(ho[r], n), rng.choice(ho[r], n)
            delay = np.exp(rng.uniform(np.log(0.2), np.log(100.0), n))       # µs, log-uniform
            for i, j, dl in zip(a, b, delay):
                sh = int(round(dl / dt))
                N = max(rec.length[i], sh + rec.length[j])                     # the record ends with the later pulse's record
                w = tm.continued(rec, i, base, amp, t50, D, model, N, noise, rng)
                w[sh:] += tm.continued(rec, j, base, amp, t50, D, model, N - sh, noise, rng)
                waves.append(np.round(w))
                truth.append((amp[i], t50[i] + D, amp[j], t50[j] + D + sh))
            delays.append(delay)
            singles = rng.choice(ho[r], min(n, 3000), replace=False)
            swaves += [rec.wave(i, base[i]) for i in singles]
            s_truth += [(amp[i], t50[i] + D) for i in singles]
            del rec
        delay, truth, s_truth = np.concatenate(delays), np.array(truth), np.array(s_truth)
        n = len(waves)
        print(f"{n:,} synthetic pairs + {len(swaves):,} single pulses, {cfg.WORKERS} workers")
        fits = tm.fit_many(waves + swaves, model_path, cfg.WORKERS, True, cfg.MAX_TEMP)
        pf, sf = fits[:n], fits[n:]

        # match fitted pulses to the true ones
        found = np.zeros((n, 2), bool); dA = np.full((n, 2), np.nan); dT = np.full((n, 2), np.nan); nfit = np.zeros(n, int)
        for q, (nt, pk, _) in enumerate(pf):
            nfit[q] = len(pk)
            used = set()
            for u, (At, pt) in enumerate(((truth[q, 0], truth[q, 1]), (truth[q, 2], truth[q, 3]))):
                best = None
                for r_, (A_, p_) in enumerate(pk):
                    if r_ not in used and abs(p_ - pt) <= TOL and (best is None or abs(p_ - pt) < abs(pk[best][1] - pt)):
                        best = r_
                if best is not None:
                    used.add(best)
                    dA[q, u] = pk[best][0] / At - 1; dT[q, u] = (pk[best][1] - pt) * dt * 1e3
                    found[q, u] = abs(dA[q, u]) <= AMP_TOL
        rsd = lambda v: float(1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v)))) if np.isfinite(v).sum() > 10 else np.nan
        sA = np.array([pk[0][0] / At - 1 if len(pk) == 1 else np.nan for (_, pk, _), (At, _) in zip(sf, s_truth)])
        sT = np.array([(pk[0][1] - pt) * dt * 1e3 if len(pk) == 1 else np.nan for (_, pk, _), (_, pt) in zip(sf, s_truth)])
        single = dict(n=len(swaves), one_pulse_fraction=float(np.mean([len(pk) == 1 for _, pk, _ in sf])),
                      sd_amp_pct=100 * rsd(sA), bias_amp_pct=100 * float(np.nanmedian(sA)),
                      sd_time_ns=rsd(sT), bias_time_ns=float(np.nanmedian(sT)))
        edges = np.array([0.2, 0.5, 1, 1.5, 2, 3, 5, 8, 12, 20, 30, 45, 70, 100])
        c = np.sqrt(edges[1:] * edges[:-1]); bi = np.digitize(delay, edges) - 1
        rows = []
        for k_ in range(len(c)):
            q = bi == k_
            rows.append(dict(delay_us=float(c[k_]), n=int(q.sum()), eff_1st=float(found[q, 0].mean()),
                             eff_2nd=float(found[q, 1].mean()), eff_both=float(found[q].all(1).mean()),
                             sd_amp_1st_pct=100 * rsd(dA[q, 0]), sd_amp_2nd_pct=100 * rsd(dA[q, 1]),
                             sd_time_1st_ns=rsd(dT[q, 0]), sd_time_2nd_ns=rsd(dT[q, 1]),
                             extra_pulses_per_record=float(np.mean(np.maximum(nfit[q] - 2, 0)))))
        # the second pulse is harder to find when it is small: efficiency split at half the median amplitude
        small_cut = 0.5 * s["median_amplitude"]
        by_amp = []
        for lo, hi in ((1, 2.5), (2.5, 4), (4, 10), (10, 100)):
            q = (delay >= lo) & (delay < hi)
            sm, bg = q & (truth[:, 2] < small_cut), q & (truth[:, 2] >= small_cut)
            by_amp.append(dict(delay_from_us=lo, delay_to_us=hi, small_second_found=float(found[sm, 1].mean()),
                               n_small=int(sm.sum()), large_second_found=float(found[bg, 1].mean()), n_large=int(bg.sum())))
        np.savez(os.path.join(md, "benchmark.npz"), delay=delay, truth=truth, found=found, dA=dA, dT=dT, nfit=nfit, sA=sA, sT=sT)
        tab = pd.DataFrame(rows)
        tab.to_csv(os.path.join(md, "benchmark.csv"), index=False)
        print(tab.round(3).to_string(index=False))
        print(f"single pulses: {single}")
        sm2 = [found[(bi == k_) & (truth[:, 2] < small_cut), 1].mean() if np.any((bi == k_) & (truth[:, 2] < small_cut))
               else np.nan for k_ in range(len(c))]
        lg2 = [found[(bi == k_) & (truth[:, 2] >= small_cut), 1].mean() for k_ in range(len(c))]
        plots.benchmark(md, c, tab.eff_1st, tab.eff_2nd, tab.eff_both, tab.sd_amp_1st_pct, tab.sd_amp_2nd_pct,
                        tab.sd_time_1st_ns, tab.sd_time_2nd_ns, single, sm2, lg2, small_cut)
        update(md, "benchmark", dict(single=single, by_delay=rows, by_second_amplitude=by_amp, small_cut_adc=small_cut,
                                     tolerance_samples=TOL, amplitude_tolerance=AMP_TOL))
