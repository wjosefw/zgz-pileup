"""Template-matching (TM) model: settings, templates and fitting of records.

The fitter (fitter.py) describes every pulse by "point models": for each offset k from the
pulse's reference point, the sample value is A * (a0 + a1 A + a2 A^2), and a record is fitted as
a sum of such pulses. Here:
  * pulses are aligned on their 50 % rise crossing; the reference point is D samples later
    (the centre of the flat top), because the maximum sample of these flat-topped 8-bit pulses
    is ill-defined;
  * the head of the template (offsets below the tail join) is fitted per offset over clean
    pulses (quadratic in A);
  * after the join the template is the measured undisturbed tail (tail.py), linear in A, out to
    the longest record, so the undershoot of every pulse is included;
  * fast filter length L, trigger threshold, trigger-to-reference distance and the residual
    threshold for adding pulses are measured on the clean pulses (characterise()).
"""
from __future__ import annotations

import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from .fitter import Parameters, PointModels, WaveformFitter, compute_fastfilter, find_triggers

PRE_SAMPLES = 16      # baseline samples before t50 inside the template range
ETA0 = 0.7            # chi2/ndf improvement needed to add a pulse (fitter default)
MULTI_MAX = 0.002     # largest fraction of clean pulses allowed to give >1 trigger
L_CANDIDATES = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]


def triggers(w, L, threshold):
    """Fast-filter triggers, minus those in the last L samples: there the filter's look-ahead
    half runs past the record end (zeros), so a tail below baseline fakes a rising edge."""
    tr = find_triggers(compute_fastfilter(w, L), threshold)
    return tr[tr < len(w) - L]


def characterise(waves, A, t50, shape, ref, D, t10_minus_t50, rng):
    """Fast-filter length and trigger threshold from clean pulses (lists of baseline-subtracted
    recorded samples). Returns settings and the scans for the figures."""
    S = np.nan_to_num(shape)
    n = min(len(waves), 20000)
    # noise: std of the fast filter of (pulse - A * shape) on the tail, where only noise is left
    resid = []
    for x, a, t in zip(waves[:3000], A[:3000], t50[:3000]):
        j = np.arange(len(x)) + ref - t
        T = np.where((j >= 0) & (j < len(S)), S[np.clip(j, 0, len(S) - 1)], 0)
        resid.append(x - a * T)
    sig, ffmax = [], []
    for L in L_CANDIDATES:
        s_ = [compute_fastfilter(r, L)[120:len(r) - 2 * L].std() for r in resid if len(r) - 2 * L > 140]
        sig.append(np.median(s_)); ffmax.append(compute_fastfilter(S, L).max())
    sig, ffmax = np.array(sig), np.array(ffmax)
    snr = ffmax / sig
    onset = t50[:n] + t10_minus_t50
    L_max = int(np.percentile(onset, 1)) - 3
    allowed = [L for L in L_CANDIDATES if L <= L_max]
    if not allowed:
        raise RuntimeError(f"pre-trigger too short for any fast-filter length (onset 1st pct {np.percentile(onset, 1)})")
    L = allowed[int(np.argmax([snr[L_CANDIDATES.index(l)] for l in allowed]))]
    sigL = sig[L_CANDIDATES.index(L)]
    # threshold scan in units of the filter noise
    ths = np.round(np.arange(2.0, 12.01, 0.25) * sigL, 4)
    ffs = [compute_fastfilter(x, L) for x in waves[:n]]
    counts = []
    for th in ths:
        nt = np.array([np.sum(tr < len(f) - L) for f in ffs for tr in [find_triggers(f, th)]])
        counts.append([np.mean(nt == 0), np.mean(nt == 1), np.mean(nt > 1)])
    counts = np.array(counts)
    ok = np.nonzero(counts[:, 2] <= MULTI_MAX)[0]
    th = float(ths[ok[0]] if len(ok) else ths[-1])
    first = np.array([(tr[0] if len(tr) else -1) for x in waves[:n] for tr in [triggers(x, L, th)]])
    good = first >= 0
    peak2trig = int(np.round(np.median((t50[:n] + D)[good] - first[good])))
    scans = dict(L=L_CANDIDATES, ff_noise=sig.tolist(), ff_peak_unit=ffmax.tolist(), snr=snr.tolist(), L_max=L_max,
                 thresholds=ths.tolist(), frac_zero=counts[:, 0].tolist(), frac_one=counts[:, 1].tolist(),
                 frac_multi=counts[:, 2].tolist())
    return dict(L=int(L), ff_threshold=th, ff_noise=float(sigL), threshold_sigma=th / sigL, peak2trig=peak2trig,
                smallest_triggering_amplitude=th / ffmax[L_CANDIDATES.index(L)]), scans


def head_coefficients(X, A, rangeleft, K_end):
    """Per-offset quadratic fit of x/A vs A. X: (n, K_end - rangeleft) aligned so column 0 is
    offset rangeleft. Returns (K_end - rangeleft, 3)."""
    G = np.stack([np.ones_like(A), A, A ** 2], 1).astype(np.float64)
    return np.linalg.lstsq(G, (X / A[:, None]).astype(np.float64), rcond=None)[0].T


def build_coefficients(head, rangeleft, K0, blend, tail_k, tail_y, K1):
    """Full point-model table on offsets [-length, length): head up to K0, blended linearly into
    the measured tail over `blend` samples, tail up to K1."""
    length = K1 + 1
    coeffs = np.zeros((2 * length, 3))
    k_head = np.arange(rangeleft, rangeleft + len(head))
    coeffs[k_head + length] = head
    kt = np.arange(K0, K1 + 1)
    yt = np.interp(kt, tail_k, tail_y)
    w = np.clip((kt - K0) / max(blend, 1), 0, 1)[:, None]
    h = np.zeros((len(kt), 3))
    inside = kt < rangeleft + len(head)
    h[inside] = head[kt[inside] - rangeleft]
    coeffs[kt + length] = (1 - w) * h + w * np.c_[yt, np.zeros_like(yt), np.zeros_like(yt)]
    return coeffs, length


class Model:
    """A TM model: point-model coefficients plus settings, saved as one .npz."""

    def __init__(self, coeffs, settings):
        self.coeffs, self.s = coeffs, settings
        p = Parameters(length=settings["length"], rangeleft=settings["rangeleft"], rangeright=settings["rangeright"],
                       L=settings["L"], noise=settings.get("noise", 0.01), eta0=settings["eta0"],
                       peak2trig=settings["peak2trig"], max_A=settings["max_A"],
                       saturation_level=settings["saturation"])
        pm = PointModels(coeffs=coeffs, length=p.length, rangeleft=p.rangeleft, rangeright=p.rangeright)
        self.fitter = WaveformFitter(pm, p)

    def save(self, path):
        np.savez(path, coeffs=self.coeffs, settings=json.dumps(self.s))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["coeffs"], json.loads(str(z["settings"])))

    def unit_pulse(self, A=None):
        """(offsets, template value per unit amplitude) at amplitude A (default: median clean amplitude)."""
        A = self.s["median_amplitude"] if A is None else A
        L = self.s["length"]
        k = np.arange(-L, L)
        y = self.coeffs @ np.array([1, A, A * A])
        m = (k >= self.s["rangeleft"]) & (k <= self.s["rangeright"])
        return k[m], y[m]

    def fit(self, w, residual_search=True):
        """Fit one baseline-subtracted record. Returns (n_triggers, [(A, pos), ...], chi2/ndf)."""
        p = self.fitter.params
        tr = triggers(w, p.L, self.s["ff_threshold"])
        if len(tr) == 0:
            return 0, [], np.nan
        left, right = max(0, tr[0] + p.rangeleft), min(len(w), tr[-1] + p.rangeright)
        try:
            par, chi2 = self.fitter.fit_window(w, tr.tolist(), left, right, npeaks0=len(tr),
                                               skip_residual_search=not residual_search)
        except Exception:
            return len(tr), [], np.nan
        return len(tr), [(float(par[2 * i]), float(par[2 * i + 1])) for i in range(len(par) // 2)], float(chi2)

    def synthesize(self, n, pulses):
        return self.fitter._synthesize(np.arange(n, dtype=float), np.ravel(pulses)) if len(pulses) else np.zeros(n)


# ------------------------------------------------------------------ parallel fitting
_model = None
_resid = True


def _init(path, residual_search):
    global _model, _resid
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    _model = Model.load(path)
    _resid = residual_search


def _fit(w):
    return _model.fit(w, _resid)


def cpu_temperature():
    """Hottest CPU sensor (°C) from /sys/class/hwmon, NaN if not available."""
    t = []
    for f in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
        try:
            name = open(os.path.join(os.path.dirname(f), "name")).read().strip()
            if name in ("coretemp", "k10temp", "zenpower", "cpu_thermal"):
                t.append(int(open(f).read()) / 1000)
        except OSError:
            pass
    return max(t) if t else float("nan")


def fit_many(waves, model_path, workers=4, residual_search=True, max_temp=85.0, chunk=2000, verbose=True):
    """Fit a list of records with `workers` processes. Between chunks, pauses while the CPU is
    hotter than max_temp (if the temperature can be read)."""
    out = []
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(model_path, residual_search)) as pool:
        for c in range(0, len(waves), chunk):
            if cpu_temperature() > max_temp:
                print(f"  CPU above {max_temp:.0f} C, pausing", flush=True)
                while cpu_temperature() > max_temp - 15:
                    time.sleep(10)
            out += pool.map(_fit, waves[c:c + chunk], chunksize=16)
            if verbose:
                done = min(c + chunk, len(waves))
                el = time.time() - t0
                print(f"  fitted {done:,}/{len(waves):,}  {el / 60:.1f} min elapsed, "
                      f"~{el / done * (len(waves) - done) / 60:.0f} min left", flush=True)
    return out


def continued(rec, i, base, amp, t50, D, model, n, noise, rng):
    """Clean record i, continued after its end with amp * (unit pulse) plus noise, to n samples."""
    x = rec.wave(i, base[i])
    out = np.empty(n)
    m = min(len(x), n)
    out[:m] = x[:m]
    if n > m:
        k, y = model.unit_pulse()
        kk = np.arange(m, n) - (t50[i] + D)
        out[m:] = amp[i] * np.interp(kk, k, y, left=0, right=0) + rng.normal(0, noise, n - m)
    return out


N_TRAIN = 250_000      # clean pulses used to fit the templates
N_CHARACTERISE = 20_000


def training_selection(train_idx, rng, n_train=N_TRAIN):
    """Training pulses pooled over runs, proportionally to their clean pulses.
    train_idx: {run: clean, not held-out record numbers}. Returns ({run: sorted records},
    [(run, record), ...] subset for the fast-filter characterisation)."""
    n_all = sum(len(v) for v in train_idx.values())
    tr = {r: np.sort(rng.choice(v, min(len(v), int(round(n_train * len(v) / n_all))), replace=False))
          for r, v in train_idx.items()}
    flat = [(r, int(i)) for r, v in tr.items() for i in v]
    sub = [flat[j] for j in rng.permutation(len(flat))[:N_CHARACTERISE]]
    return tr, sub
