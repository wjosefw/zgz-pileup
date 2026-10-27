"""The pulse tail and undershoot, measured from the baseline of the next record.

Clean records end before the pulse has returned to baseline (record length grows with
amplitude; the photopeak records stop ~45 µs after the pulse while the tail is still ~+4 ADC),
so the tail and the undershoot that follows it are not in the clean records. They are measured
from pairs of consecutive records: when a record starts while the previous pulse is still in
its tail, its pre-trigger baseline equals that pulse's tail at that moment. Using pairs where
the previous record is a single clean pulse of known amplitude A, with nothing before it for
ISOLATION µs, baseline / A against the time since that pulse traces the tail of an
undisturbed pulse, out to several hundred µs.

The time offset of the baseline window from the previous pulse's reference point is
    k = (t_i - t_prev) / dt + (N_BASELINE - 1) / 2 - (t50_prev + D)
in samples, which assumes both out.txt time stamps refer to the same sample of their records
(the curve then joins the clean-pulse shape and crosses zero where pile-up fits put it).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .pulses import N_BASELINE

ISOLATION_US = 1000.0
AMIN, AMAX = 15.0, 40.0
ZERO_GAP_US = 5000.0
BIN = 8               # samples per bin (1 µs at 128 ns)
MIN_PAIRS = 100       # pairs per bin to keep it


def baseline_pairs(times, length, cls, base, amp, t50, D, dt_us, max_gap_us):
    """(k in samples after the previous pulse's reference point, baseline / A_prev, zero point)."""
    idx = np.nonzero(np.isfinite(times))[0]
    gap = np.diff(times[idx]) * 1e6
    prev_gap = np.r_[np.inf, gap[:-1]]
    cur, prv = idx[1:], idx[:-1]
    zero = float(np.median(base[cur[gap > ZERO_GAP_US]]))
    sel = (cls[prv] == 0) & np.isfinite(amp[prv]) & (amp[prv] >= AMIN) & (amp[prv] <= AMAX) & \
          (prev_gap > ISOLATION_US) & (gap <= max_gap_us) & (cur == prv + 1) & \
          (gap >= length[prv] * dt_us)                      # the records do not overlap
    c, p, g = cur[sel], prv[sel], gap[sel]
    k = g / dt_us + (N_BASELINE - 1) / 2 - (t50[p] + D)
    return k, (base[c] - zero) / amp[p], zero


def tail_curve(k, y):
    """1-µs-binned median of the pairs, Savitzky-Golay smoothed. Returns bin centres (samples),
    smoothed curve, raw medians, their errors and pair counts."""
    edges = np.arange(np.floor(k.min() / BIN) * BIN, k.max() + BIN, BIN)
    cen, med, err, cnt = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        q = (k >= lo) & (k < hi)
        if q.sum() >= MIN_PAIRS:
            cen.append(k[q].mean()); med.append(np.median(y[q]))
            err.append(1.2533 * y[q].std() / np.sqrt(q.sum())); cnt.append(q.sum())
    cen, med, err, cnt = map(np.array, (cen, med, err, cnt))
    smooth = savgol_filter(med, 9, 2) if len(med) >= 9 else med
    return cen, smooth, med, err, cnt
