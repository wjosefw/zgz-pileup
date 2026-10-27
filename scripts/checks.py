"""Independent checks of a finished analysis, to run and read yourself. Nothing is changed.

1. Records and out.txt rows. Counts the records of c<ch> and the rows of out.txt, compares the
   energy of record k with row k (naive pairing) and with the row given by step 1, and lists
   where the row numbering slips.
2. The fitter on records with known pulses. Builds records from held-out clean pulses (never
   used for the templates) placed at chosen delays, fits them exactly as step 8 does, and prints
   the true and fitted pulses side by side. Figure: models/.../figures/13_checks_fitter.png.
3. Energies of resolved pile-up. Photopeak of clean pulses and of pulses resolved from pile-up
   records, from events.csv: if the fits were biased, the two peaks would differ.

    python scripts/checks.py [--runs RUN1] [--channels 1]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zgzpileup import io, pulses, tm  # noqa: E402
from zgzpileup.project import banner, load_npz, need, parse  # noqa: E402

TOL = 0.05             # energy agreement between a record and a row (out.txt units)
# check 2: (delay of each pulse after the first, in µs; amplitude of each as a fraction of the photopeak)
CASES = [((0,), (1.0,)), ((0, 20), (1.0, 1.0)), ((0, 6), (1.0, 0.5)), ((0, 3), (1.0, 1.0)),
         ((0, 1.5), (1.0, 1.0)), ((0, 8, 15), (1.0, 0.6, 1.0)), ((0, 4), (0.3, 1.0)), ((0, 30), (1.0, 0.2))]

proj, _ = parse(__doc__.split("\n")[0])
for run, folder in proj.runs.items():
    for ch in proj.channels:
        d = proj.run_dir(run, ch)
        md = proj.model_dir(proj.group_of(run), ch)
        need(d, "records.npz", "times.npz", "pulses.npz")

        # ---------------------------------------------------------------- 1
        banner(f"{run} channel {ch} — check 1: records and out.txt rows")
        fmax = np.load(os.path.join(d, "records.npz"))["fmax"]
        E = io.read_out_txt(os.path.join(folder, "out.txt"), ch).E.to_numpy()
        row = np.load(os.path.join(d, "times.npz"))["out_row"]
        print(f"records in c{ch}: {len(fmax):,}    rows of channel {ch} in out.txt: {len(E):,}    "
              f"difference: {len(fmax) - len(E):+,}")
        n = min(len(fmax), len(E))
        naive = np.abs(fmax[:n] - E[:n]) < TOL
        ok = row >= 0
        paired = np.abs(fmax[ok] - E[row[ok]]) < TOL
        off = row[ok] - np.nonzero(ok)[0]
        slips = np.nonzero(np.diff(off))[0]
        edges = [0] + [int(np.nonzero(ok)[0][s]) + 1 for s in slips] + [n]
        print("record k paired with row k (naive), energy agrees in:")
        for lo, hi in zip(edges[:-1], edges[1:]):
            print(f"   records {lo:>10,} – {hi - 1:>10,}: {naive[lo:hi].mean():6.1%}")
        print(f"paired by step 1 (row = record + offset): {ok.sum():,} records ({ok.mean():.3%}), "
              f"energy agrees in {paired.mean():.1%}")
        print(f"offset slips after records: {[int(np.nonzero(ok)[0][s]) for s in slips]} "
              f"(offsets {sorted(set(off.tolist()))}); records without a row: {int((~ok).sum())}, "
              f"first at {int(np.nonzero(~ok)[0][0]) if (~ok).any() else '-'}")

        # ---------------------------------------------------------------- 2
        banner(f"{run} channel {ch} — check 2: the fitter on records with known pulses")
        need(md, "tm_model.npz")
        model = tm.Model.load(os.path.join(md, "tm_model.npz"))
        s = model.s
        rec = io.Records.load(os.path.join(d, "records.npz"))
        P = load_npz(os.path.join(d, "pulses.npz"))
        base, amp, t50 = P["baseline"].astype(float), P["amp"], P["t50"]
        D, dt = s["D"], s["dt_us"]
        a_peak = pulses.photopeak(amp[P["cls"] == 0])[0]
        ho = np.nonzero(P["holdout"])[0]
        rng = np.random.default_rng(proj.seed + 7)
        fig, axs = plt.subplots(2, 4, figsize=(17, 6.5))
        print(f"{'case':>4}  {'true pulses (µs, ADC)':<44} {'fitted pulses (µs, ADC)':<44}")
        for c, ((delays, fracs), ax) in enumerate(zip(CASES, axs.ravel())):
            # for each pulse, a held-out clean pulse with an amplitude close to the requested one
            picks = [ho[np.argmin(np.abs(amp[ho] - f * a_peak) + rng.uniform(0, 1.5, len(ho)))] for f in fracs]
            shifts = [int(round(dl / dt)) for dl in delays]
            N = max(rec.length[i] + sh for i, sh in zip(picks, shifts)) + 50
            w = np.zeros(N)
            for i, sh in zip(picks, shifts):
                w[sh:] += tm.continued(rec, i, base, amp, t50, D, model, N - sh, 0.0, rng)
            w = np.round(w + rng.normal(0, 0.6, N))
            truth = [((t50[i] + D + sh) * dt, amp[i]) for i, sh in zip(picks, shifts)]
            _, pk, chi2 = model.fit(w)
            fitted = sorted([(p * dt, A) for A, p in pk])
            fmt = lambda v: ", ".join(f"({t:.2f}, {a:.1f})" for t, a in v)
            print(f"{c + 1:>4}  {fmt(truth):<44} {fmt(fitted):<44} χ²/ndf {chi2:.2f}")
            t = np.arange(N) * dt
            ax.plot(t, w, color="0.45", lw=0.7, label="record")
            ax.plot(t, model.synthesize(N, [(A, p) for A, p in pk]), color="#eb6834", lw=1.1, label="TM fit")
            for tt, a in truth:
                ax.axvline(tt, color="#2a78d6", ls=":", lw=0.9)
            ax.set_xlim(0, min(t[-1], max(tt for tt, _ in truth) + 40))
            ax.set_title(f"case {c + 1}: {len(truth)} true, {len(pk)} fitted", fontsize=10)
            ax.set_xlabel("µs")
        axs[0, 0].legend(fontsize=8)
        fig.suptitle("Records built from held-out clean pulses (dotted: true positions) and their TM fits")
        fig.tight_layout()
        os.makedirs(os.path.join(md, "figures"), exist_ok=True)
        fig.savefig(os.path.join(md, "figures", "13_checks_fitter.png"), dpi=130)
        plt.close(fig)
        print("positions are the reference point (flat-top centre) of each pulse")

        # ---------------------------------------------------------------- 3
        banner(f"{run} channel {ch} — check 3: photopeak of clean and of resolved pulses")
        if os.path.exists(os.path.join(d, "events.csv")):
            ev = pd.read_csv(os.path.join(d, "events.csv"), usecols=["kind", "n_pulses", "amplitude_adc", "flag"])
            ev = ev[ev.flag.isna()]
            for name, sel in (("clean", ev.kind == "clean"),
                              ("resolved pile-up (≥ 2 pulses)", (ev.kind == "fitted") & (ev.n_pulses >= 2))):
                p, fw = pulses.photopeak(ev[sel].amplitude_adc.to_numpy())
                print(f"{name:<32} {sel.sum():>10,} pulses   photopeak {p:6.2f} ADC   FWHM {100 * fw / p:.1f} %")
        else:
            print("no events.csv yet (run step 8)")
