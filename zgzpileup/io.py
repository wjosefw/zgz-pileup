"""Reading the acquisition files.

c<ch>.txt (or c<ch>.zip containing it): one triggered record per block, blocks separated by a
blank line, three columns per sample:
    time since the start of the record (s) | height | VST filter output
The height is an ADC value multiplied by a constant step (e.g. 254.4); it is stored here as
integer ADC counts (int16). The maximum of the VST column equals the energy column of out.txt
for the same record, which is how records are matched to out.txt.

out.txt: one row per record of every channel: channel, time since the start of the run (s),
energy (VST filter maximum).
"""
from __future__ import annotations

import os
import time
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Records:
    """All records of one channel, concatenated."""
    adc: np.ndarray      # int16, every sample of every record
    offset: np.ndarray   # int64, start of record i in adc
    length: np.ndarray   # int32, samples in record i
    fmax: np.ndarray     # float64, maximum of the VST filter column of record i
    dt_us: float         # sampling period (µs)
    adc_step: float      # height units per ADC count

    def __len__(self):
        return len(self.length)

    def wave(self, i, baseline=0.0):
        """Record i as float, minus `baseline`."""
        return self.adc[self.offset[i]:self.offset[i] + self.length[i]].astype(float) - baseline

    def save(self, path):
        np.savez(path, adc=self.adc, offset=self.offset, length=self.length, fmax=self.fmax,
                 dt_us=self.dt_us, adc_step=self.adc_step)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["adc"], z["offset"], z["length"], z["fmax"], float(z["dt_us"]), float(z["adc_step"]))


def _open_channel_file(path):
    """File object for c<ch>.txt, or for the .txt inside c<ch>.zip."""
    if path.endswith(".zip"):
        z = zipfile.ZipFile(path)
        names = [n for n in z.namelist() if n.endswith(".txt")]
        if len(names) != 1:
            raise ValueError(f"{path}: expected one .txt inside, found {names}")
        return z.open(names[0])
    return open(path, "rb")


def _adc_step(h):
    """Height units per ADC count, from the spacing of the distinct height values."""
    u = np.unique(np.abs(h[np.isfinite(h)]))
    u = u[(u > 0) & (u < 40 * u[u > 0].min())]
    step0 = np.median(np.diff(np.r_[0.0, u]))
    n = np.round(u / step0)
    return float(np.sum(n * u) / np.sum(n * n))


def read_channel(path, chunk_lines=5_000_000, verbose=True):
    """Read c<ch>.txt / c<ch>.zip into Records. About 3 min per 25 GB file."""
    t0 = time.time()
    f = _open_channel_file(path)
    reader = pd.read_csv(f, sep=r"\s+", header=None, names=["t", "h", "f"], skip_blank_lines=False,
                         chunksize=chunk_lines, engine="c", dtype=np.float64)
    adc_parts, lengths, fmaxs = [], [], []
    carry_h = np.empty(0); carry_f = np.empty(0)
    step = dt_us = None
    n_lines = 0
    for chunk in reader:
        t = chunk["t"].to_numpy(); h = chunk["h"].to_numpy(); fv = chunk["f"].to_numpy()
        n_lines += len(h)
        if step is None:
            step = _adc_step(h[:1_000_000])
            d = np.diff(t[:100_000]); d = d[np.isfinite(d) & (d > 0)]
            dt_us = float(np.round(np.median(d) * 1e6, 6))
        blank = np.isnan(h)
        h = np.r_[carry_h, h]; fv = np.r_[carry_f, fv]; blank = np.r_[np.zeros(len(carry_h), bool), blank]
        seg = np.cumsum(blank)               # block id of every line
        keep = ~blank
        hv, fvv, sv = h[keep], fv[keep], seg[keep]
        # the block still open at the end of the chunk is carried over to the next one
        if not blank[-1]:
            done = sv < seg[-1]
            carry_h, carry_f = hv[~done], fvv[~done]
        else:
            done = np.ones(len(sv), bool)
            carry_h = np.empty(0); carry_f = np.empty(0)
        hv, fvv, sv = hv[done], fvv[done], sv[done]
        if len(sv):
            starts = np.r_[0, np.nonzero(np.diff(sv))[0] + 1]
            lengths.append(np.diff(np.r_[starts, len(sv)]))
            fmaxs.append(np.maximum.reduceat(fvv, starts))
            adc_parts.append(np.round(hv / step).astype(np.int16))
        if verbose:
            print(f"  {n_lines / 1e6:7.0f} M lines, {sum(len(x) for x in lengths):,} records, "
                  f"{time.time() - t0:5.0f} s", flush=True)
    if len(carry_h):
        lengths.append(np.array([len(carry_h)])); fmaxs.append(np.array([carry_f.max()]))
        adc_parts.append(np.round(carry_h / step).astype(np.int16))
    length = np.concatenate(lengths).astype(np.int32)
    offset = np.r_[0, np.cumsum(length[:-1], dtype=np.int64)]
    rec = Records(np.concatenate(adc_parts), offset, length, np.concatenate(fmaxs), dt_us, step)
    if verbose:
        print(f"{len(rec):,} records, {rec.adc.size:,} samples; sampling {dt_us * 1e3:.1f} ns; "
              f"ADC step {step:.2f}; {time.time() - t0:.0f} s")
    return rec


def read_out_txt(path, channel):
    """Rows of out.txt for one channel: DataFrame with columns t (s) and E, in file order."""
    out = pd.read_csv(path, sep=r"\s+", header=None, names=["ch", "t", "E"])
    return out[out.ch == channel].reset_index(drop=True)


def align_times(fmax, E, tol=0.05, run=10, window=60, max_shift=200):
    """Pair records with out.txt rows, in order; returns the out.txt row of every record
    (-1 when unmatched) and counts.

    Row = record + offset, and the offset only changes where a record has no row or a row has
    no record, which is rare. The energies identify the offset: where fmax[i] == E[i + offset]
    the pair is certain (an anchor). Many pairs differ in energy (small or truncated pulses;
    in some channels a large fraction), so a mismatch alone does not mean the offset changed;
    a change shows up as a run of `run` consecutive mismatches. Then the new offset is the shift
    within +-max_shift that matches most records in the next `window`. Records between two
    anchors with the same offset are paired by position whatever their energy; records
    between anchors of different offsets are paired only if their energy matches one of them.
    """
    n, m = len(fmax), len(E)
    anchor_i, anchor_o = [], []
    o, i = 0, 0
    changes = 0
    while i < n:
        hi = min(n, i + 200_000)
        idx = np.arange(i, hi)
        jj = idx + o
        ok = (jj >= 0) & (jj < m)
        match = np.zeros(len(idx), bool)
        match[ok] = np.abs(fmax[idx[ok]] - E[jj[ok]]) < tol
        # first run of `run` consecutive mismatches
        miss = ~match
        c = np.convolve(miss.astype(int), np.ones(run, int), mode="valid")
        bad = np.nonzero(c == run)[0]
        stop = bad[0] if len(bad) else len(idx)
        good = np.nonzero(match[:stop])[0]
        anchor_i.extend(idx[good]); anchor_o.extend([o] * len(good))
        if not len(bad):
            i = hi
            continue
        # offset change after the last anchor before the run
        start = idx[good[-1]] + 1 if len(good) else idx[stop]
        lo, up = start, min(n, start + window)
        best, best_score = None, 0
        for d in range(-max_shift, max_shift + 1):
            j0 = lo + o + d
            if j0 < 0 or j0 + (up - lo) > m or d == 0:
                continue
            sc = int(np.sum(np.abs(fmax[lo:up] - E[j0:j0 + up - lo]) < tol))
            if sc > best_score or (sc == best_score and best is not None and abs(d) < abs(best)):
                best, best_score = d, sc
        if best is None or best_score < 3:
            i = idx[stop] + run           # nothing consistent: skip past the run and keep the offset
            continue
        o += best; changes += 1
        i = lo
    ai, ao = np.array(anchor_i, np.int64), np.array(anchor_o, np.int64)
    row = np.full(n, -1, np.int64)
    row[ai] = ai + ao
    # fill between consecutive anchors
    seg_same = ao[1:] == ao[:-1]
    for a, b, oo, same in zip(ai[:-1], ai[1:], ao[:-1], seg_same):
        if b - a > 1:
            between = np.arange(a + 1, b)
            if same:
                row[between] = between + oo
            else:
                for q in between:
                    for off in (oo, ao[np.searchsorted(ai, b)]):
                        if 0 <= q + off < m and abs(fmax[q] - E[q + off]) < tol:
                            row[q] = q + off
    row[(row < 0) | (row >= m)] = -1
    stats = dict(anchors=int(len(ai)), offset_changes=changes, energy_mismatch_pairs=int(np.sum(row >= 0) - len(ai)),
                 final_offset=int(o))
    return row, stats


def find_file(run_dir, channel, names):
    for n in names:
        p = os.path.join(run_dir, n.format(ch=channel))
        if os.path.exists(p):
            return p
    return None
