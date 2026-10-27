"""Figures written by the pipeline (all to <work>/figures)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

BLUE, ORANGE, AQUA, YELLOW, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlelocation": "left",
                     "axes.titlesize": 10, "legend.frameon": False})


def _save(fig, work, name):
    d = os.path.join(work, "figures"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def _grid(ax):
    for a in np.ravel(ax):
        a.grid(color=GRID, lw=0.5)


def gaps(work, gap_us):
    fig, a = plt.subplots(figsize=(7, 3.6))
    a.hist(np.clip(gap_us, 1, 1e8), np.logspace(0, 8, 161), color=INK2)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("time between consecutive records (µs)"); a.set_ylabel("records")
    a.set_title("Time between consecutive records")
    _grid(a)
    return _save(fig, work, "01_record_gaps.png")


def selection(work, rec, F, cls, band, classes, k, dt, rng):
    """Area vs amplitude with the band, and examples of what the selection trigger sees."""
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    a = ax[0]
    show = (cls != 2) & (F["amp"] > 0)
    a.hist2d(F["amp"][show], F["area"][show], bins=[250, 250],
             range=[[0, np.percentile(F["amp"][show], 99.95)], [0, np.percentile(F["area"][show], 99.95)]],
             norm=LogNorm(), cmap="Greys")
    xx = np.linspace(band["amp"].min(), band["amp"].max(), 200)
    sp = np.interp(xx, band["amp"], band["spread"])
    a.plot(xx, np.polyval(band["coef"], xx), color=ORANGE, lw=1.2, label="single pulses (quadratic)")
    a.plot(xx, np.polyval(band["coef"], xx) + k * sp, color=ORANGE, ls=":", lw=1, label=f"upper limit (+{k:g}σ)")
    a.set_xlabel("amplitude (ADC)"); a.set_ylabel("area (ADC × samples)"); a.legend(fontsize=8)
    a.set_title("(a) all records: area vs amplitude")
    a = ax[1]
    rel = band["relative_area"]
    for c_, col in ((5, YELLOW), (6, AQUA), (4, ORANGE), (0, BLUE)):
        q = np.nonzero((cls == c_) & (F["amp"] > 0))[0]
        if len(q):
            q = rng.choice(q, min(len(q), 30000), replace=False)
            a.scatter(F["amp"][q], rel[q], s=1, alpha=0.3, color=col, label=classes[c_], rasterized=True)
    a.plot(xx, 1 + k * sp / np.polyval(band["coef"], xx), color=INK, ls=":", lw=0.9, label="upper limit")
    a.axhline(1, color=INK, lw=0.5)
    a.set_ylim(0.2, 2.0); a.set_xlabel("amplitude (ADC)"); a.set_ylabel("area / single-pulse curve")
    a.set_title("(b) relative area by class (the lower group: second pulse shape)"); a.legend(fontsize=8, markerscale=8)
    a = ax[2]
    from .fitter import compute_fastfilter
    pool = np.nonzero((cls == 4) & (F["amp"] > 15))[0]
    for q in rng.permutation(pool)[:200]:      # an example whose second trigger falls in the first 15 µs
        x = rec.wave(q, F["base"][q])
        f = compute_fastfilter(x, 4)
        above = np.nonzero(f >= F["sel_threshold"])[0]
        if len(above) and np.any((above > above[0] + 25) & (above < 15 / dt)):
            break
    t = np.arange(len(x)) * dt
    a.plot(t, x, color=INK2, lw=0.8, label="record (ADC)")
    a.plot(t, compute_fastfilter(x, 4), color=ORANGE, lw=0.9, label="selection filter (L = 4)")
    a.axhline(F["sel_threshold"], color=ORANGE, ls=":", lw=0.8, label=f"threshold {F['sel_threshold']:.1f} ADC")
    a.set_xlim(0, min(t[-1], 25)); a.set_xlabel("µs"); a.legend(fontsize=8, loc="center right")
    a.set_title(f"(c) a record with a second trigger (record {q})")
    _grid(ax); fig.tight_layout()
    return _save(fig, work, "02_clean_selection.png")


def classes_gallery(work, rec, cls, base, classes, dt, rng, n_per=6):
    rows = [c for c in range(len(classes)) if np.any(cls == c)]
    fig, ax = plt.subplots(len(rows), n_per, figsize=(16, 2.1 * len(rows)), squeeze=False)
    for row, c in zip(ax, rows):
        pool = np.nonzero(cls == c)[0]
        for a, i in zip(row, rng.choice(pool, min(n_per, len(pool)), replace=False)):
            x = rec.wave(i, base[i])
            a.plot(np.arange(len(x)) * dt, x, color=BLUE if c == 0 else ORANGE, lw=0.8)
            a.text(0.98, 0.94, f"rec {i}", transform=a.transAxes, ha="right", va="top", fontsize=7, color=INK2)
        row[0].set_title(f"{classes[c]}: {len(pool):,} records ({100 * len(pool) / len(cls):.1f} %)")
        row[0].set_ylabel("ADC")
    for a in ax[-1]:
        a.set_xlabel("µs")
    _grid(ax); fig.tight_layout()
    return _save(fig, work, "03_record_classes.png")


def clean_overview(work, batch, amp_b, shape, ref, length_c, amp_c, fmax, fmax_c, dt):
    t = np.arange(batch.shape[1]) * dt
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    a = ax[0, 0]
    for j in range(min(40, len(batch))):
        a.plot(t, batch[j], color=BLUE, lw=0.6, alpha=0.6)
    a.set_title("(a) 40 clean pulses (after the record end: amplitude × mean shape)")
    a.set_xlabel("µs since record start"); a.set_ylabel("ADC")
    a = ax[0, 1]
    y = batch / amp_b[:, None]
    H, xe, ye = np.histogram2d(np.tile(t, len(batch)), y.ravel(), bins=[batch.shape[1], 200],
                               range=[[0, t[-1]], [-0.3, 1.3]])
    a.pcolormesh(xe, ye, H.T, norm=LogNorm(), cmap="Blues", rasterized=True)
    a.set_title(f"(b) {len(batch):,} clean pulses / amplitude (density)")
    a.set_xlabel("µs since record start"); a.set_ylabel("x / A")
    a = ax[1, 0]
    a.hist2d(amp_c, length_c, bins=[120, 120], range=[[0, np.percentile(amp_c, 99.9) * 1.3], [length_c.min(), length_c.max()]],
             norm=LogNorm(), cmap="Blues")
    a.set_title("(c) recorded length vs amplitude (clean)"); a.set_xlabel("amplitude (ADC)"); a.set_ylabel("samples")
    a = ax[1, 1]
    fmax, fmax_c = fmax[np.isfinite(fmax)], fmax_c[np.isfinite(fmax_c)]
    bins = np.linspace(0, np.percentile(fmax, 99.9) * 1.1, 400)
    a.hist(fmax, bins, histtype="step", color=INK2, label="all records")
    a.hist(fmax_c, bins, histtype="step", color=BLUE, lw=1.2, label="clean")
    a.set_yscale("log"); a.legend(); a.set_xlabel("VST filter maximum (out.txt energy)")
    a.set_title("(d) VST energy, all vs clean records")
    _grid(ax); fig.tight_layout()
    return _save(fig, work, "04_clean_pulses.png")


def tail(work, k, y, cen, smooth, med, err, head_k, head_y, model_k, model_y, K0, dt):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    a = ax[0]
    a.scatter(k * dt, y, s=2, color=INK2, alpha=0.2, rasterized=True, label=f"record pairs ({len(k):,})")
    a.plot(cen * dt, smooth, color=BLUE, lw=1.4, label="tail curve (1 µs bins, smoothed)")
    a.plot(head_k * dt, head_y, color=ORANGE, lw=1.2, label="clean-pulse shape")
    a.axhline(0, color=INK, lw=0.5)
    a.set_ylim(-0.3, 0.6); a.set_xlim(head_k[0] * dt, cen[-1] * dt)
    a.set_xlabel("µs after the pulse's reference point"); a.set_ylabel("value / amplitude")
    a.set_title("(a) next-record baseline after an isolated clean pulse"); a.legend(fontsize=8)
    a = ax[1]
    n = len(med)   # cen may be longer: extended beyond the last measured bin
    a.errorbar(cen[:n] * dt, med, err, color=BLUE, fmt=".", ms=3, lw=0.6, label="tail curve (bin medians)")
    if len(cen) > n:
        a.plot(cen[n - 1:] * dt, smooth[n - 1:], color=BLUE, ls=":", lw=1, label="extrapolated")
    if model_k is not None:                 # the template exists after 05_train_tm.py
        a.plot(model_k * dt, model_y, color=INK, lw=1.3, label="TM template (head + tail)")
    else:
        a.plot(head_k * dt, head_y, color=ORANGE, lw=1.2, label="clean-pulse shape")
    a.axvline(K0 * dt, color=INK2, ls=":", lw=0.8); a.text(K0 * dt, 0.2, " join", color=INK2, fontsize=8)
    a.axhline(0, color=INK, lw=0.5); a.set_xlim(0, cen[-1] * dt if model_k is None else model_k[-1] * dt)
    a.set_ylim(-0.25, 1.05)
    a.set_xlabel("µs after the reference point"); a.set_title("(b) the full unit pulse used by TM"); a.legend(fontsize=8)
    _grid(ax); fig.tight_layout()
    return _save(fig, work, "05_tail_undershoot.png")


def tm_settings(work, scans, s, model, dt):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    a = ax[0]
    Ls = np.array(scans["L"]); snr = np.array(scans["snr"])
    a.plot(Ls, snr / snr.max(), "o-", color=BLUE, label="trigger SNR (unit pulse)")
    a.plot(Ls, np.array(scans["ff_noise"]) / max(scans["ff_noise"]), "s-", color=INK2, label="filter noise")
    a.axvspan(scans["L_max"] + 0.5, Ls[-1] * 1.1, color=INK2, alpha=0.08)
    a.text(scans["L_max"] + 1, 0.1, " pre-trigger\n too short", fontsize=8, color=INK2)
    a.axvline(s["L"], color=ORANGE, ls="--"); a.set_xscale("log"); a.set_xticks(Ls); a.set_xticklabels(Ls)
    a.set_xlabel("fast-filter length L (samples)"); a.set_title(f"(a) fast-filter length: L = {s['L']}"); a.legend(fontsize=8)
    a = ax[1]
    th = np.array(scans["thresholds"])
    a.plot(th, scans["frac_one"], color=BLUE, label="exactly 1 trigger")
    a.plot(th, scans["frac_zero"], color=INK2, label="no trigger")
    a.plot(th, scans["frac_multi"], color=ORANGE, label="> 1 trigger")
    a.axvline(s["ff_threshold"], color=ORANGE, ls="--"); a.set_yscale("log"); a.set_ylim(1e-5, 1.5)
    a.set_xlabel("threshold (ADC)"); a.set_ylabel("fraction of clean pulses")
    a.set_title(f"(b) trigger threshold: {s['ff_threshold']:.2f} ADC ({s['threshold_sigma']:.1f} σ)"); a.legend(fontsize=8)
    a = ax[2]
    if model is None:                       # before the templates exist (04_tm_params.py)
        a.set_axis_off()
        a.text(0.5, 0.5, "templates: after 05_train_tm.py", ha="center", va="center", color=INK2)
        _grid(ax[:2]); fig.tight_layout()
        return _save(fig, work, "06_tm_settings.png")
    for A, c in zip((10, 30, 60, 110), (AQUA, BLUE, ORANGE, INK)):
        k, y = model.unit_pulse(A)
        a.plot(k * dt, A * y, color=c, lw=1.1, label=f"A = {A} ADC")
    a.axhline(0, color=INK, lw=0.5)
    a.set_xlabel("µs from the reference point"); a.set_ylabel("ADC"); a.set_title("(c) templates"); a.legend(fontsize=8)
    _grid(ax); fig.tight_layout()
    return _save(fig, work, "06_tm_settings.png")


def calibration(work, fit, peak):
    fig, a = plt.subplots(figsize=(8, 4.4))
    e = fit[0] * peak["amp"] + fit[1]
    bins = np.linspace(0, np.percentile(e, 99.9) * 1.1, 500)
    a.hist(e, bins, histtype="step", color=BLUE, label="clean pulses")
    a.axvline(peak["kev"], color=ORANGE, ls=":", lw=1, label=f"photopeak, set to {peak['kev']:.0f} keV")
    if np.isfinite(peak["sum_kev"]):
        a.axvline(peak["sum_kev"], color=INK2, ls=":", lw=1,
                  label=f"sum peak at {peak['sum_kev']:.0f} keV = {peak['sum_kev'] / peak['kev']:.3f} × photopeak")
    a.set_yscale("log"); a.set_xlabel("energy (keV)"); a.set_ylabel("pulses / bin"); a.legend(fontsize=8)
    a.set_title(f"E = {fit[0]:.3f} keV/ADC × A; photopeak FWHM {peak['fwhm_pct']:.1f} %")
    _grid(a); fig.tight_layout()
    return _save(fig, work, "07_energy_calibration.png")


def benchmark(work, c, eff1, eff2, both, sdA1, sdA2, sdt1, sdt2, single, small2, large2, cut):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    a = ax[0]
    a.plot(c, eff1, "o-", color=BLUE, label="1st pulse found"); a.plot(c, eff2, "s-", color=ORANGE, label="2nd pulse found")
    a.plot(c, both, "^-", color=INK2, label="both found")
    a.plot(c, large2, ":", color=ORANGE, lw=1.4, label=f"2nd pulse found, A ≥ {cut:.0f} ADC")
    a.plot(c, small2, "--", color=ORANGE, lw=1.0, label=f"2nd pulse found, A < {cut:.0f} ADC")
    a.set_xscale("log"); a.set_ylim(0, 1.02)
    a.set_xlabel("delay between the pulses (µs)"); a.set_ylabel("fraction"); a.set_title("(a) detection"); a.legend(fontsize=8)
    a = ax[1]
    a.plot(c, sdA1, "o-", color=BLUE, label="1st pulse"); a.plot(c, sdA2, "s-", color=ORANGE, label="2nd pulse")
    a.axhline(single["sd_amp_pct"], color=INK2, ls=":", label="single pulses")
    a.set_xscale("log"); a.set_xlabel("delay (µs)"); a.set_ylabel("%")
    a.set_title("(b) energy error: robust std of A_fit / A_true − 1"); a.legend(fontsize=8)
    a = ax[2]
    a.plot(c, sdt1, "o-", color=BLUE, label="1st pulse"); a.plot(c, sdt2, "s-", color=ORANGE, label="2nd pulse")
    a.axhline(single["sd_time_ns"], color=INK2, ls=":", label="single pulses")
    a.set_xscale("log"); a.set_xlabel("delay (µs)"); a.set_ylabel("ns")
    a.set_title("(c) time error: robust std of t_fit − t_true"); a.legend(fontsize=8)
    _grid(ax); fig.tight_layout()
    return _save(fig, work, "08_benchmark.png")


def fits_gallery(work, name, title, waves, fits, ids, model, dt, cols=4):
    rows = int(np.ceil(len(waves) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(4.4 * cols, 2.9 * rows), squeeze=False)
    for a, w, (nt, pk, chi2), i in zip(ax.flat, waves, fits, ids):
        t = np.arange(len(w)) * dt
        a.plot(t, w, color=INK2, lw=0.6)
        if pk:
            a.plot(t, model.synthesize(len(w), pk), color=ORANGE, lw=1.2)
            for A, p in pk:
                if 0 <= p < len(w):
                    a.plot(t, model.synthesize(len(w), [(A, p)]), color=BLUE, lw=0.6, ls="--")
        inside = sum(0 <= p < len(w) for _, p in pk)
        a.set_title(f"rec {i}: {inside} pulses, χ²/ndf {chi2:.2f}", fontsize=8)
    for a in ax.flat[len(waves):]:
        a.axis("off")
    ax.flat[0].plot([], [], color=INK2, label="data"); ax.flat[0].plot([], [], color=ORANGE, label="TM fit")
    ax.flat[0].plot([], [], color=BLUE, ls="--", label="single pulses"); ax.flat[0].legend(fontsize=7)
    for a in ax[-1]:
        a.set_xlabel("µs")
    fig.suptitle(title); _grid(ax); fig.tight_layout()
    return _save(fig, work, name)


def spectra(work, ev):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    has_e = np.isfinite(ev["energy_kev"]).any()
    col = "energy_kev" if has_e else "amplitude_adc"
    v = ev[col].to_numpy()
    hi = np.nanpercentile(v, 99.9) * 1.1
    bins = np.linspace(0, hi, 500)
    groups = (("clean", BLUE, (ev.kind == "clean")),
              ("resolved pile-up (records with ≥ 2 pulses)", ORANGE, (ev.kind == "fitted") & (ev.n_pulses >= 2)),
              ("fitted single pulses", YELLOW, (ev.kind == "fitted") & (ev.n_pulses == 1)),
              ("on an earlier tail", AQUA, (ev.kind == "baseline")))
    for kind, c, sel in groups:
        q = sel & ev.flag.isna()
        if q.any():
            for a in ax:
                a.hist(v[q], bins, histtype="step", color=c, lw=1.1, density=a is ax[1],
                       label=f"{kind} ({q.sum():,} pulses)")
    ax[0].set_yscale("log"); ax[0].set_title("(a) spectra (counts)")
    ax[1].set_yscale("log"); ax[1].set_title("(b) spectra (area-normalised)")
    for a in ax:
        a.set_xlabel("energy (keV)" if has_e else "amplitude (ADC)"); a.legend(fontsize=8)
    _grid(ax); fig.tight_layout()
    return _save(fig, work, "11_energy_spectra.png")


def time_distributions(work, ev):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    t = ev.time_s.to_numpy()
    ok = np.isfinite(t) & ev.flag.isna()
    order = np.argsort(t[ok])
    ts = t[ok][order]
    kinds = np.where((ev.kind == "fitted") & (ev.n_pulses >= 2), "pileup", ev.kind).astype(object)[ok][order]
    dtp = np.diff(ts) * 1e6
    a = ax[0]
    bins = np.logspace(-1, 8, 181)
    a.hist(dtp, bins, histtype="step", color=INK, label="all pulses")
    for kind, c, lab in (("clean", BLUE, "clean pulse"), ("pileup", ORANGE, "pulse in a pile-up record")):
        q = kinds[1:] == kind
        a.hist(dtp[q], bins, histtype="step", color=c, label=lab)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("time since the previous pulse (µs)")
    a.set_title("(a) time to the previous pulse"); a.legend(fontsize=8)
    a = ax[1]
    pu = ev[(ev.kind == "fitted") & (ev.n_pulses >= 2) & ev.flag.isna()]
    within = []
    for _, g in pu.groupby("record"):
        if len(g) > 1:
            within.append(np.diff(np.sort(g.time_in_record_us.to_numpy())))
    within = np.concatenate(within) if within else np.array([])
    a.hist(within, np.linspace(0, 160, 161), color=ORANGE)
    a.set_xlabel("µs between resolved pulses in the same record"); a.set_title("(b) spacing of resolved pile-up pulses")
    a = ax[2]
    hours = (ts - ts[0]) / 3600
    hb = np.arange(0, hours.max() + 1, max(1, hours.max() / 200))
    for kind, c in (("clean", BLUE), ("pileup", ORANGE)):
        q = kinds == kind
        h, e = np.histogram(hours[q], hb)
        a.plot(0.5 * (e[1:] + e[:-1]), h / np.diff(e) / 3600, color=c, lw=0.8, label=kind)
    a.set_yscale("log"); a.set_xlabel("hours since the first pulse"); a.set_ylabel("pulses / s")
    a.set_title("(c) rate vs time"); a.legend(fontsize=8)
    _grid(ax); fig.tight_layout()
    return _save(fig, work, "12_time_distributions.png")
