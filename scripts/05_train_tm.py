"""Step 5. Train the TM templates.

At every offset k from the reference point (flat-top centre), the template value is a quadratic
in the amplitude, h_k(A) = a0 + a1 A + a2 A^2, fitted over 250,000 training pulses (clean, not
held out, pooled over the runs of the group). Where the records have ended the training pulses
are continued with their amplitude times the mean shape; after the start of the measured tail
(step 3) the template is the tail itself, out past the longest record. The threshold for adding
pulses from the fit residual is the 95th percentile of the largest residual of single fits of
held-out clean pulses.

  models/<group>/ch<c>/tm_model.npz      templates and settings (used by steps 7 and 8)
  models/<group>/ch<c>/tm_settings.json  the same settings, readable

Figures: models/.../figures/05_tail_undershoot.png, 06_tm_settings.png

    python scripts/05_train_tm.py [--channels ...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zgzpileup import io, plots, pulses, tm  # noqa: E402
from zgzpileup.project import banner, load_npz, need, parse, update  # noqa: E402

BLEND = 32             # samples over which the template head blends into the measured tail
N_NOISE = 400          # held-out clean pulses for the residual threshold

proj, _ = parse(__doc__.split("\n")[0])
for ch in proj.channels:
    for group, runs in proj.groups().items():
        banner(f"channel {ch}: TM templates")
        md = proj.model_dir(group, ch)
        need(md, "shape.npz", "tail.npz", "tm_params.json")
        S = np.load(os.path.join(md, "shape.npz"))
        T = np.load(os.path.join(md, "tail.npz"))
        prm = json.load(open(os.path.join(md, "tm_params.json")))
        shape, ref, sat, dt = S["shape"], int(S["ref"]), int(S["saturation"]), float(S["dt_us"])
        D = pulses.shape_times(shape, ref)["D"]
        P = {r: load_npz(os.path.join(proj.run_dir(r, ch), "pulses.npz")) for r in runs}
        rng = np.random.default_rng(proj.seed)
        tr, _ = tm.training_selection({r: np.nonzero((P[r]["cls"] == 0) & ~P[r]["holdout"])[0] for r in runs}, rng)

        # training matrix: each training pulse from rangeleft to K_end samples around its reference point
        rangeleft = -(D + tm.PRE_SAMPLES)
        K0, K1 = int(T["K0"]), int(T["K1"])
        K_end = K0 + BLEND
        X, A = [], []
        for r in runs:
            rec = io.Records.load(os.path.join(proj.run_dir(r, ch), "records.npz"))
            base, amp, t50 = P[r]["baseline"].astype(float), P[r]["amp"], P[r]["t50"]
            p = t50[tr[r]] + D
            L_out = int(p.max()) + K_end + 1
            for c in range(0, len(tr[r]), 50_000):
                part = tr[r][c:c + 50_000]
                Xp = pulses.padded_batch(rec, part, base, amp[part], t50, shape, ref, L_out)
                cols = p[c:c + 50_000, None] + np.arange(rangeleft, K_end)[None, :]
                X.append(np.where(cols >= 0, np.take_along_axis(Xp, np.clip(cols, 0, None), 1), 0).astype(np.float32))
            A.append(amp[tr[r]].astype(np.float64))
            del rec
        X, A = np.concatenate(X), np.concatenate(A)
        print(f"fitting the templates on {len(A):,} pulses")
        head = tm.head_coefficients(X, A, rangeleft, K_end)
        del X
        coeffs, length = tm.build_coefficients(head, rangeleft, K0, BLEND, T["cen"], T["smooth"], K1)
        clean_amp = np.concatenate([P[r]["amp"][P[r]["cls"] == 0] for r in runs])
        settings = dict(prm["settings"], D=int(D), ref=ref, rangeleft=int(rangeleft), rangeright=int(K1), length=int(length),
                        eta0=tm.ETA0, max_A=float(1.2 * sat), saturation=float(sat), noise=0.01,
                        median_amplitude=float(np.median(clean_amp)), dt_us=dt, K0=K0, K1=K1, blend=BLEND,
                        n_train=int(len(A)), runs=runs)
        model = tm.Model(coeffs, settings)

        # residual threshold for adding pulses, from single fits of held-out clean pulses
        ho = {r: np.nonzero(P[r]["holdout"])[0] for r in runs}
        n_ho = sum(len(v) for v in ho.values())
        mx, ratio = [], []
        for r in runs:
            rec = io.Records.load(os.path.join(proj.run_dir(r, ch), "records.npz"))
            pick = rng.choice(ho[r], min(len(ho[r]), int(round(N_NOISE * len(ho[r]) / n_ho))), replace=False)
            for i in pick:
                w = rec.wave(i, P[r]["baseline"][i])
                nt, pk, _ = model.fit(w, residual_search=False)
                if nt == 1 and pk:
                    mx.append(float((model.synthesize(len(w), pk) - w).max())); ratio.append(pk[0][0] / P[r]["amp"][i])
            del rec
        settings["noise"] = float(np.percentile(mx, 95))
        model = tm.Model(coeffs, settings)
        model.save(os.path.join(md, "tm_model.npz"))
        json.dump(dict(settings=settings, scans=prm["scans"]), open(os.path.join(md, "tm_settings.json"), "w"), indent=1)

        plots.tm_settings(md, prm["scans"], settings, model, dt)
        fin = np.isfinite(shape)
        mk, my = model.unit_pulse()
        plots.tail(md, T["k"], T["y"], T["cen"], T["smooth"], T["med"], T["err"],
                   (np.arange(len(shape)) - ref - D)[fin], shape[fin], mk, my, K0, dt)
        res = {k_: settings[k_] for k_ in ("L", "ff_threshold", "threshold_sigma", "ff_noise", "peak2trig", "D", "rangeleft",
                                           "rangeright", "noise", "smallest_triggering_amplitude", "n_train")}
        res["single_fit_amplitude_ratio_tm_over_template"] = float(np.median(ratio))
        print(f"templates from offset {rangeleft} to {K1} samples; residual threshold {settings['noise']:.2f} ADC; "
              f"TM / template amplitude on single pulses {np.median(ratio):.4f}")
        update(md, "tm", res)
