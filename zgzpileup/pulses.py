"""Selection of the clean pulses, the mean pulse shape and pulse amplitudes.

Every record is put in one class (first rule that applies):
  saturated     some sample reaches the ADC ceiling
  baseline      the pre-trigger samples are not flat around zero (|mean| > 1.5 ADC or
                peak-to-peak > 3 ADC): the record starts on the tail of an earlier pulse
  no pulse      the selection trigger does not fire
  2nd trigger   the selection trigger fires again after the pulse's own rise: pile-up
  slow rise     10-90 % rise longer than MAX_RISE samples: pile-up within the rise
  off diagonal  area above the single-pulse band (or below AREA_FLOOR of it)
  clean         none of the above: one undisturbed pulse

The selection trigger is the fast filter of the TM (tm.py) with a short window, SEL_L samples,
so that it re-arms quickly after a pulse's rise; its threshold is SEL_SIGMA times the noise of
the filter on the pre-trigger samples. Triggers within MERGE samples of the first one belong to
the same rise, and triggers in the last SEL_L samples are ignored (there the filter's look-ahead
window runs past the record end).

The band: single pulses lie on a smooth curve of area against amplitude (the area is summed
over the recorded samples, and records end earlier for small pulses, so the curve is slightly
concave). The curve is followed outward from the photopeak along the main population (band_fit)
and fitted with a quadratic. Pile-up adds area, so records more than BAND_K robust standard
deviations above the curve are rejected. Records below it are kept: the data contain a second
population of single pulses that decay faster (less area for the same amplitude; a few % of the
pulses in one channel, a quarter to a third in others), and both shapes go into the templates
together. relative_area (area / curve) tells the two apart. Pile-up that the trigger cannot see
(two pulses within a few µs, whose areas and amplitudes add up) is not removed by the band,
which is why both checks are needed.

The pulse shape is the median of clean pulses aligned on their 50 % rise crossing (t50) and
divided by their amplitude. The amplitude of a clean pulse is the least-squares scale of that
shape over the recorded samples ("template amplitude").
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit

from .fitter import compute_fastfilter, find_triggers

CLASSES = ["clean", "baseline", "saturated", "no pulse", "2nd trigger", "slow rise", "off diagonal"]
N_BASELINE = 12      # pre-trigger samples used for the baseline
BASE_MAX_ABS = 1.5   # ADC
BASE_MAX_PTP = 3     # ADC
SEL_L = 4            # selection trigger: fast-filter length (samples)
SEL_SIGMA = 5.5      # selection trigger: threshold in units of the filter noise
MERGE = 20           # samples: triggers this close to the first belong to the same rise
MAX_RISE = 20        # samples
W = 3                # moving average used for the amplitude and the rise
BAND_K = 3.0         # band: records more than BAND_K robust standard deviations ABOVE the curve are rejected
AREA_FLOOR = 0.3     # and records with less than this fraction of the curve's area


def selection_noise(rec, base, ptp):
    """Standard deviation (5-sigma clipped) of the short fast filter on flat pre-trigger samples."""
    flat = np.nonzero((np.abs(base) <= BASE_MAX_ABS) & (ptp <= BASE_MAX_PTP))[0]
    sel = flat[:: max(1, len(flat) // 100_000)]
    v = np.concatenate([compute_fastfilter(rec.adc[rec.offset[i]:rec.offset[i] + N_BASELINE] - base[i], SEL_L)
                        [SEL_L:N_BASELINE - SEL_L + 1] for i in sel])
    s = v.std()
    for _ in range(3):
        s = v[np.abs(v) < 5 * s].std()
    return float(s)


def record_features(rec, verbose=True):
    """Per record: baseline, its peak-to-peak, maximum, area, smoothed amplitude, 10-90 % rise,
    t50 and the number of selection triggers."""
    n, off, ln = len(rec), rec.offset, rec.length
    pre = rec.adc[off[:, None] + np.arange(N_BASELINE)].astype(np.float64)
    base = pre.mean(1); ptp = pre.max(1) - pre.min(1)
    mx = np.maximum.reduceat(rec.adc, off)
    seg = np.repeat(np.arange(n), ln)
    area = np.bincount(seg, rec.adc.astype(np.float64), minlength=n) - base * ln
    noise = selection_noise(rec, base, ptp)
    thr = SEL_SIGMA * noise
    amp = np.zeros(n, np.float32); rise = np.zeros(n, np.int32); t50 = np.full(n, -1, np.int32)
    ntr = np.zeros(n, np.int16)
    for i in range(n):
        x = rec.adc[off[i]:off[i] + ln[i]].astype(np.float64) - base[i]
        tr = find_triggers(compute_fastfilter(x, SEL_L), thr)
        tr = tr[tr < len(x) - SEL_L]
        ntr[i] = 0 if len(tr) == 0 else 1 + int(np.sum(tr - tr[0] >= MERGE))
        s = np.convolve(x, np.ones(W) / W, mode="valid")
        A = s.max()
        amp[i] = A
        if A > 0:
            i10, i50, i90 = (int(np.argmax(s >= f * A)) for f in (0.1, 0.5, 0.9))
            rise[i] = i90 - i10; t50[i] = i50 + W // 2
        if verbose and (i + 1) % 500_000 == 0:
            print(f"  features {i + 1:,}/{n:,}", flush=True)
    return dict(base=base.astype(np.float32), ptp=ptp, mx=mx, area=area, amp=amp, rise=rise, t50=t50, ntr=ntr,
                sel_noise=noise, sel_threshold=thr)


def band_fit(amp, area, cand, win=0.10, min_n=20):
    """The single-pulse curve of area vs amplitude, followed outward from the most populated
    amplitude bin: in each bin only records whose area/amplitude is within `win` of the value
    predicted from the neighbouring bins are used (median and robust spread), so the curve stays
    on the main population even where a second population, or pile-up, outnumbers it. A
    quadratic is then fitted to the bin medians."""
    a, r, ar = amp[cand], area[cand] / amp[cand], area[cand]
    hi = np.percentile(a, 99.99)
    edges = np.unique(np.r_[np.arange(np.floor(a.min()), 40, 1.0), np.arange(40, hi + 5, 5.0)])
    cen = 0.5 * (edges[1:] + edges[:-1])
    ratio = np.full(len(cen), np.nan); mid = np.full(len(cen), np.nan)
    med = np.full(len(cen), np.nan); spread = np.full(len(cen), np.nan); nn = np.zeros(len(cen))

    def fill(k, pred):
        q = (a >= edges[k]) & (a < edges[k + 1])
        sel = np.abs(r[q] / pred - 1) < win
        if sel.sum() < min_n:
            return False
        v = ar[q][sel]
        ratio[k] = np.median(r[q][sel]); mid[k] = np.median(a[q][sel]); med[k] = np.median(v)
        spread[k] = 1.4826 * np.median(np.abs(v - med[k])); nn[k] = sel.sum()
        return True

    start = int(np.argmax(np.histogram(a, edges)[0]))
    q = (a >= edges[start]) & (a < edges[start + 1])
    fill(start, np.median(r[q]))
    for step in (1, -1):
        done, k = [start], start + step
        while 0 <= k < len(cen):
            last = [j for j in done if np.isfinite(ratio[j])][-4:]
            pred = np.polyval(np.polyfit(mid[last], ratio[last], 1), cen[k]) if len(last) >= 2 else ratio[last[-1]]
            if fill(k, pred):
                done.append(k)
            k += step
    ok = np.isfinite(med)
    coef = np.polyfit(mid[ok], med[ok], 2, w=np.sqrt(nn[ok]))
    return dict(coef=coef, amp=mid[ok], median=med[ok], spread=spread[ok], n=nn[ok])


def classify(F, saturation):
    """Class of every record from its features; returns the classes and the band."""
    n = len(F["base"])
    cls = np.zeros(n, np.int8)
    base_ok = (np.abs(F["base"]) <= BASE_MAX_ABS) & (F["ptp"] <= BASE_MAX_PTP)
    rules = [(F["mx"] >= saturation, 2), (~base_ok, 1), (F["ntr"] == 0, 3), (F["ntr"] >= 2, 4), (F["rise"] > MAX_RISE, 5)]
    undecided = np.ones(n, bool)
    for cond, k in rules:
        m = undecided & cond
        cls[m] = k; undecided &= ~m
    cand = undecided & (F["amp"] > 0)
    band = band_fit(F["amp"], F["area"], cand)
    centre = np.polyval(band["coef"], F["amp"])
    width = BAND_K * np.interp(F["amp"], band["amp"], band["spread"])
    # one-sided: pile-up adds area; records below the curve are single pulses that decay faster
    # (a second pulse shape) and are kept, so the templates describe both shapes together
    off = cand & ((F["area"] > centre + width) | (F["area"] < AREA_FLOOR * centre))
    cls[off] = 6
    band["relative_area"] = np.where(F["amp"] > 0, F["area"] / np.maximum(centre, 1e-9), np.nan).astype(np.float32)
    return cls, band
def aligned_stack(rec, sel, base, amp, t50, ref, L):
    """Records sel as rows of x / A, shifted so t50 sits at `ref`; NaN where a record has no sample."""
    acc = np.full((len(sel), L), np.nan, np.float32)
    for k, i in enumerate(sel):
        x = rec.wave(i, base[i]) / amp[i]
        s = ref - t50[i]
        lo, hi = max(0, s), min(L, len(x) + s)
        acc[k, lo:hi] = x[lo - s:hi - s]
    return acc


def shape_from_stack(acc):
    """Median of the aligned rows, normalised to a maximum of 1; NaN where fewer than 50 rows reach."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # columns no record reaches: NaN, expected
        T = np.nanmedian(acc, axis=0)
        count = np.sum(np.isfinite(acc), axis=0)
    T[count < 50] = np.nan
    return T / np.nanmax(T), count


def aligned_shape(rec, idx, base, amp, t50, ref, L, n=200_000, seed=0):
    """Median of x / A over clean records, shifted so t50 sits at `ref`; NaN where no record reaches."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(idx, min(n, len(idx)), replace=False)
    return shape_from_stack(aligned_stack(rec, sel, base, amp, t50, ref, L))


def template_amplitudes(rec, idx, base, t50, shape, ref):
    """Least-squares scale of the t50-aligned shape over each record's recorded samples."""
    L = len(shape)
    S = np.nan_to_num(shape)
    A = np.empty(len(idx), np.float32)
    for k, i in enumerate(idx):
        x = rec.wave(i, base[i])
        s = ref - t50[i]                 # record sample j <-> shape index j + s
        j = np.arange(len(x)); m = (j + s >= 0) & (j + s < L)
        T = S[j[m] + s]
        A[k] = np.dot(x[m], T) / np.dot(T, T)
    return A


def padded_batch(rec, idx, base, amp, t50, shape, ref, L_out):
    """Clean records as a (n, L_out) array, record sample j at index j, continued after the
    record end with amp * shape (the records stop while the tail is still above baseline)."""
    S = np.nan_to_num(shape)
    out = np.zeros((len(idx), L_out), np.float32)
    for k, i in enumerate(idx):
        x = rec.wave(i, base[i])
        m = min(len(x), L_out)
        out[k, :m] = x[:m]
        if m < L_out:
            j = np.arange(m, L_out) + ref - t50[i]
            ok = (j >= 0) & (j < len(S))
            out[k, m:][ok] = amp[k] * S[j[ok]]
    return out


def shape_times(shape, ref):
    """Rise, flat-top centre (the reference point, D samples after t50), decay and zero crossing."""
    T = np.nan_to_num(shape)
    pk = int(np.argmax(np.convolve(T, np.ones(9) / 9, mode="same")))
    i10 = int(np.argmax(T >= 0.1)); i90 = int(np.argmax(T >= 0.9))
    after = T[pk:]
    dec10 = pk + int(np.argmax(after < 0.1))
    return dict(rise_10_90=i90 - i10, t10_minus_t50=i10 - ref, D=pk - ref, decay10_from_peak=dec10 - pk)


def photopeak(a):
    """Position and FWHM of the main peak of an amplitude spectrum (Gaussian fit to its top)."""
    h, e = np.histogram(a, np.linspace(0, np.percentile(a, 99.5), 600))
    c = 0.5 * (e[1:] + e[:-1])
    up = c > 0.4 * c[np.argmax(h)]
    mu = c[up][np.argmax(h[up])]
    g = lambda x, n, m, s: n * np.exp(-0.5 * ((x - m) / s) ** 2)
    sel = np.abs(c - mu) < 0.06 * mu
    p = [h[sel].max(), mu, 0.03 * mu]
    for _ in range(3):
        p, _ = curve_fit(g, c[sel], h[sel], p0=p)
        sel = np.abs(c - p[1]) < 1.2 * abs(p[2])
    return float(p[1]), float(2.3548 * abs(p[2]))


def peak_between(a, lo, hi):
    """Centre of the highest peak of a between lo and hi (Gaussian fit to its top), or NaN."""
    v = a[(a > lo) & (a < hi)]
    if len(v) < 200:
        return float("nan")
    h, e = np.histogram(v, np.linspace(lo, hi, 120))
    c = 0.5 * (e[1:] + e[:-1])
    hs = np.convolve(h, np.ones(5) / 5, mode="same")
    mu = c[np.argmax(hs)]
    sel = np.abs(c - mu) < 0.04 * mu
    try:
        p, _ = curve_fit(lambda x, n, m, w: n * np.exp(-0.5 * ((x - m) / w) ** 2), c[sel], h[sel],
                         p0=[h[sel].max(), mu, 0.03 * mu])
        return float(p[1])
    except RuntimeError:
        return float(mu)
