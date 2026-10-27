"""Template-matching (TM) pile-up fitter.

Python port of the LaBr3Wave template-matching code (https://github.com/pkuNucExp/LaBr3Wave,
upstream commit 0c176f5). Only the parts used here are kept: Parameters, the fast filter and
trigger search, PointModels, saturation detection and WaveformFitter. Local changes by
J. A. Avellaneda: NN-seeded triggers, per-peak bounds, max_A, saturation level, analytic
Jacobian. Upstream has no licence file: do not redistribute this file publicly
without the upstream authors' permission.

Conventions: positions are sample indices, amplitudes are in waveform units, the fit parameter
vector is [A1, pos1, A2, pos2, ...], saturated samples are left out of the residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.signal import find_peaks


@dataclass
class Parameters:
    """Algorithm parameters for LaBr3 pile-up recovery.

    Adjust these to your sampling rate and detector response. Key roles:
    - `length`: influence window half-width around a peak (samples)
    - `rangeleft`/`rangeright`: effective range used within `[-length, +length)`
    - `L`: fastfilter window length
    - `noise`: residual peak height threshold when adding peaks
    - `eta0`: baseline chi2/ndf improvement factor for accepting a new peak
    - `peak2trig`: initial peak-to-trigger offset for seeding
    - `max_A`: upper bound on fitted amplitude; set below the cubic turnover
      of A*h(A) to prevent convergence to spurious large-A solutions
    - `saturation_level`: known digitizer/simulation clipping ceiling. When set,
      `fit_window` rejects every sample at or above this value from the fit
      instead of relying on `detect_saturation`'s auto-detection heuristic
      (which needs a run of >=3 identical samples and can miss short plateaus).
    """

    length: int = 400          # range of influence (samples)
    rangeleft: int = -100       # effective range left (relative to peak)
    rangeright: int = 300      # effective range right (relative to peak)
    L: int = 5                 # fastfilter window length
    noise: float = 0.01        # max noise amplitude for residual peak acceptance
    eta0: float = 0.7          # chi2/ndf improvement baseline
    peak2trig: int = 30        # approx distance from peak to fastfilter trigger
    max_A: float = 10         # upper bound on fitted amplitude; set below the cubic turnover
                               # of A*h(A) to prevent convergence to spurious large-A solutions
    saturation_level: Optional[float] = None  # known clipping ceiling; None = auto-detect only


def compute_fastfilter(wave: ArrayLike, L: int) -> np.ndarray:
    """Compute the fastfilter by subtracting adjacent box averages.

    Args:
        wave: 1D waveform array.
        L: Positive window length in samples.

    Returns:
        1D array of fastfilter values (same length as `wave`).
    """
    wave = np.asarray(wave, dtype=float)
    if L <= 0:
        raise ValueError("L must be > 0")
    # Efficient implementation using convolution of step windows
    kernel_pos = np.concatenate([np.ones(L), np.zeros(L)])
    kernel_neg = np.concatenate([np.zeros(L), np.ones(L)])
    ff = (np.convolve(wave, kernel_pos, mode="same") - np.convolve(wave, kernel_neg, mode="same")) / L
    return ff


def find_triggers(fastfilter: ArrayLike, threshold: float) -> np.ndarray:
    """Find indices where fastfilter crosses threshold with pre-conditions.

    Declares a trigger at i if ff[i] >= threshold and ff[i-1], ff[i-2] < threshold.
    """
    ff = np.asarray(fastfilter, dtype=float)
    above = ff >= threshold
    below = ff < threshold
    mask = above[2:] & below[1:-1] & below[:-2]
    return np.nonzero(mask)[0] + 2


@dataclass
class PointModels:
    """Quadratic models of normalized point height vs amplitude per offset.

    coeffs holds [a0, a1, a2] for each offset index in [0, 2*length).
    """

    coeffs: np.ndarray  # shape (nfitpnt, 3) for quadratic coefficients [a0, a1, a2]
    length: int
    rangeleft: int
    rangeright: int

    def eval(self, offset_index: int, amplitude: float) -> float:
        """Evaluate normalized height at a given offset and amplitude."""
        a0, a1, a2 = self.coeffs[offset_index]
        x = amplitude
        return a0 + a1 * x + a2 * x * x


def detect_saturation(wave: ArrayLike, known_level: Optional[float] = None) -> Tuple[float, np.ndarray]:
    """Detect saturated (clipped) samples.

    If `known_level` is given (e.g. Config.saturation_level, the digitizer/
    simulation clipping ceiling), every sample at or above it is treated as
    saturated. This is more reliable than auto-detection for short plateaus
    (fewer than 3 samples), since it doesn't depend on run length.

    Otherwise, auto-detects a saturated flat-top level as any 3 identical
    consecutive samples.

    Returns (rejval, indices_at_or_above_rejval). If not found: (-inf, empty array).
    """
    w = np.asarray(wave)
    if known_level is not None:
        idx = np.nonzero(w >= known_level)[0]
        if len(idx) == 0:
            return -np.inf, np.array([], dtype=int)
        return float(known_level), idx
    # Find first index where three consecutive samples are equal
    match = (w[1:-1] == w[:-2]) & (w[1:-1] == w[2:])
    hits = np.nonzero(match)[0]
    if len(hits) == 0:
        return -np.inf, np.array([], dtype=int)
    rejval = float(w[hits[0] + 1])
    idx = np.where(w == rejval)[0]
    return rejval, idx


class WaveformFitter:
    """Iterative multi-peak fitter using point models and residual peak search.

    Parameter vector format: [A1, pos1, A2, pos2, ...].
    Saturated samples are ignored in residuals (ROOT TF1::RejectPoint analogue).
    """

    def __init__(self, point_models: PointModels, params: Parameters, noise_level: Optional[float] = None):
        """Initialize with trained models and parameters.

        Args:
            point_models: Trained PointModels instance.
            params: Algorithm Parameters.
            noise_level: Optional override for residual peak threshold.
        """
        self.pm = point_models
        self.params = params
        self.noise = params.noise if noise_level is None else float(noise_level)

    def _model_samples_vec(self, x: np.ndarray, par: np.ndarray) -> np.ndarray:
        """Vectorized model evaluation over an array of sample positions.

        Args:
            x: 1D array of sample positions.
            par: Parameter vector [A1, pos1, A2, pos2, ...].

        Returns:
            1D array of synthesized model values at each position in `x`.
        """
        result = np.zeros(len(x), dtype=float)
        npeaks = len(par) // 2
        length = self.pm.length
        max_idx = 2 * length - 1
        coeffs = self.pm.coeffs  # (nfitpnt, 3)

        for i in range(npeaks):
            A = par[2 * i]
            pos = par[2 * i + 1]
            offsets = x - pos  # (N,)

            mask = (offsets >= self.params.rangeleft) & (offsets <= self.params.rangeright)
            if not np.any(mask):
                continue

            offs_m = offsets[mask]
            frac = offs_m + length
            x1 = np.floor(frac).astype(int)
            x2 = x1 + 1
            np.clip(x1, 0, max_idx, out=x1)
            np.clip(x2, 0, max_idx, out=x2)

            # Vectorized quadratic eval: a0 + a1*A + a2*A^2
            c1 = coeffs[x1]  # (M, 3)
            c2 = coeffs[x2]  # (M, 3)
            A_vec = np.array([1.0, A, A * A])
            y1 = c1 @ A_vec
            y2 = c2 @ A_vec

            # Linear interpolation between y1 and y2
            same = (x2 == x1)
            t = frac - np.floor(frac)
            y0 = np.where(same, y1, y1 + (y2 - y1) * t)

            result[mask] += A * y0

        return result

    def _residuals_and_jac(self, par: np.ndarray, x: np.ndarray, y: np.ndarray,
                           rejval: float, rejidx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute residuals and analytical Jacobian in a single pass.

        Returns:
            (residuals, jacobian) where jacobian has shape (len(x), len(par)).
        """
        N = len(x)
        npeaks = len(par) // 2
        length = self.pm.length
        max_idx = 2 * length - 1
        coeffs = self.pm.coeffs  # (nfitpnt, 3)

        model = np.zeros(N, dtype=float)
        jac = np.zeros((N, len(par)), dtype=float)

        for i in range(npeaks):
            A = par[2 * i]
            pos = par[2 * i + 1]
            offsets = x - pos

            mask = (offsets >= self.params.rangeleft) & (offsets <= self.params.rangeright)
            if not np.any(mask):
                continue

            frac = offsets[mask] + length
            x1 = np.floor(frac).astype(int)
            x2 = x1 + 1
            np.clip(x1, 0, max_idx, out=x1)
            np.clip(x2, 0, max_idx, out=x2)

            c1 = coeffs[x1]  # (M, 3)
            c2 = coeffs[x2]  # (M, 3)

            A_vec = np.array([1.0, A, A * A])
            y1 = c1 @ A_vec
            y2 = c2 @ A_vec

            same = (x2 == x1)
            t = frac - np.floor(frac)
            h = np.where(same, y1, y1 + (y2 - y1) * t)

            model[mask] += A * h

            # d(model)/dA_i = h + A * dh/dA
            dA_vec = np.array([0.0, 1.0, 2.0 * A])
            dy1_dA = c1 @ dA_vec
            dy2_dA = c2 @ dA_vec
            dh_dA = np.where(same, dy1_dA, dy1_dA + (dy2_dA - dy1_dA) * t)
            jac[mask, 2 * i] = h + A * dh_dA

            # d(model)/dpos_i = A * dh/dpos = -A * (y2 - y1)  [piecewise-linear deriv]
            dh_dpos = np.where(same, 0.0, -(y2 - y1))
            jac[mask, 2 * i + 1] = A * dh_dpos

        res = model - y

        # Zero out saturated samples in both residuals and Jacobian
        if np.isfinite(rejval) and len(rejidx) > 0:
            sat_mask = np.any(np.abs(x[:, None] - rejidx[None, :]) < 1, axis=1)
            res[sat_mask] = 0.0
            jac[sat_mask, :] = 0.0

        return res, jac

    def _residuals(self, par: np.ndarray, x: np.ndarray, y: np.ndarray, rejval: float, rejidx: np.ndarray) -> np.ndarray:
        """Compute model-data residuals, zeroing saturated samples."""
        res, _ = self._residuals_and_jac(par, x, y, rejval, rejidx)
        return res

    def _fit_fixed_npeaks(
        self,
        x: np.ndarray,
        y: np.ndarray,
        npeaks: int,
        vx0: Sequence[float],
        vy0: Sequence[float],
        bounds_A: Tuple[float, float],
        bounds_pos: Tuple[float, float],
        rejval: float,
        rejidx: np.ndarray,
        frac_A: Optional[float] = None,
        max_shift_pos: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, int]:
        """Least-squares fit with a fixed number of peaks; returns (par, chi2/ndf, nfev).

        If `frac_A`/`max_shift_pos` are given, each peak is bounded individually
        around its own seed value (vy0[i], vx0[i]) instead of sharing the window-wide
        `bounds_A`/`bounds_pos` range — e.g. frac_A=0.1 allows +/-10% amplitude and
        max_shift_pos=1 allows +/-1 sample position drift, per peak.
        """
        par0 = []
        for i in range(npeaks):
            par0 += [vy0[i], vx0[i]]
        par0 = np.asarray(par0, dtype=float)

        lb = []
        ub = []
        for i in range(npeaks):
            if frac_A is not None:
                lb_A = max(bounds_A[0], vy0[i] * (1.0 - frac_A))
                ub_A = min(bounds_A[1], vy0[i] * (1.0 + frac_A))
            else:
                lb_A, ub_A = bounds_A
            if max_shift_pos is not None:
                lb_pos = max(bounds_pos[0], vx0[i] - max_shift_pos)
                ub_pos = min(bounds_pos[1], vx0[i] + max_shift_pos)
            else:
                lb_pos, ub_pos = bounds_pos
            lb += [lb_A, lb_pos]
            ub += [ub_A, ub_pos]
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        def fun(p):
            r, j = self._residuals_and_jac(p, x, y, rejval, rejidx)
            self._last_jac = j
            return r
        def jac(p):
            return self._last_jac
        res = least_squares(fun, par0, jac=jac, bounds=(lb, ub), loss="linear",
                            tr_solver="exact", gtol=1e-4, xtol=1e-4, ftol=1e-4)
        ndf = max(1, len(x) - len(par0))
        chi2ndf = float(np.sum(res.fun ** 2) / ndf)
        return res.x, chi2ndf, int(res.nfev)

    def fit_window(
        self,
        wave: ArrayLike,
        vtrig: Sequence[int],
        left: int,
        right: int,
        npeaks0: int,
        plot: bool = False,
        ax: Optional["matplotlib.axes.Axes"] = None,
        show: bool = True,
        initial_amplitudes: Optional[Sequence[float]] = None,
        skip_residual_search: bool = False,
        frac_A: Optional[float] = None,
        max_shift_pos: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Fit [left, right) adding residual peaks only if chi2/ndf improves.

        Accepts a new peak if chi2/ndf decreases by a factor
        eta = 1 - (1-eta0) * (L0/L) with L0 = rangeright-rangeleft.
        Returns (best_params [A1,pos1,...], chi2ndf).

        If ``initial_amplitudes`` is provided (one per trigger), those values
        are used as starting guesses instead of the waveform value at each
        peak position.  This is useful when an external model (e.g. a NN)
        provides per-pulse amplitude estimates that are more accurate than the
        raw waveform samples — especially in pileup regions.

        If ``skip_residual_search`` is True, only the initial fit with
        ``npeaks0`` peaks is performed — the iterative residual peak search
        is skipped entirely.  Use this when the peak count is trusted (e.g.
        provided by a NN), avoiding up to 2 extra least-squares fits per
        window.

        If `plot=True`, overlays the fitted curve and annotations on a matplotlib axis.
        - Original waveform segment: gray line
        - Fitted model: red line
        - Fastfilter triggers: blue dashed vertical lines
        - Fitted peak positions: green dashed lines with markers at estimated peak height
        The plotting is a convenience feature and does not alter return values.
        """
        w = np.asarray(wave, dtype=float)
        x = np.arange(left, right, dtype=float)
        y = w[left:right]
        rejval, rejidx = detect_saturation(w, known_level=self.params.saturation_level)

        # Initial positions near triggers
        vx = [t + self.params.peak2trig for t in vtrig]
        # Initial amplitudes: use external guesses when available, else waveform samples.
        # Clamp to max_A (not 1.0) — external estimates (e.g. NN amplitudes) and pileup
        # waveform samples routinely exceed 1.0; a 1.0 clamp here would silently cap the
        # frac_A bound (vy0[i] * (1+frac_A)) for every peak above that amplitude.
        if initial_amplitudes is not None:
            vy = [min(max(float(a), 0.0), self.params.max_A) for a in initial_amplitudes]
        else:
            vy = [min(max(w[int(round(px))], 0.0), self.params.max_A) if 0 <= int(round(px)) < len(w) else 1.0 for px in vx]

        bounds_A = (0.0, self.params.max_A)
        bounds_pos = (left - self.params.rangeright, right - self.params.rangeleft)

        # First fit with npeaks0
        par, chi2ndf0, _nfev = self._fit_fixed_npeaks(x, y, npeaks0, vx[:npeaks0], vy[:npeaks0], bounds_A, bounds_pos, rejval, rejidx, frac_A, max_shift_pos)

        # Update vx, vy from fit
        vx = [par[2 * i + 1] for i in range(npeaks0)]
        vy = [par[2 * i + 0] for i in range(npeaks0)]
        npeaks = npeaks0

        # Iteratively add peaks from residuals (skipped when peak count is trusted)
        while not skip_residual_search and npeaks <= npeaks0 + 2:
            # Compute residuals on grid
            res = self._residuals(par, x, y, rejval, rejidx)
            # Find candidate peaks in residuals above noise
            peaks, props = find_peaks(res, height=self.noise)
            if len(peaks) == 0:
                break
            # Try candidates in descending order of height
            order = np.argsort(props["peak_heights"])[::-1]
            added = False
            for idx in order:
                p_idx = int(peaks[idx])
                cand_pos = float(x[p_idx])
                cand_A = float(props["peak_heights"][idx])

                # Build trial parameter vector with one new peak
                vx_trial = vx + [cand_pos]
                vy_trial = vy + [cand_A]
                par_trial, chi2ndf1, _nfev = self._fit_fixed_npeaks(
                    x, y, npeaks + 1, vx_trial, vy_trial, bounds_A, bounds_pos, rejval, rejidx,
                    frac_A, max_shift_pos,
                )

                # Acceptance criterion
                L0 = self.params.rangeright - self.params.rangeleft
                L = max(L0, right - left)
                eta = 1.0 - (1.0 - self.params.eta0) * (L0 / L)
                if chi2ndf1 < eta * chi2ndf0:
                    par = par_trial
                    chi2ndf0 = chi2ndf1
                    npeaks += 1
                    vx = [par[2 * i + 1] for i in range(npeaks)]
                    vy = [par[2 * i + 0] for i in range(npeaks)]
                    added = True
                    break

            if not added:
                break

        # Optional plotting of the fitted result
        if plot:
            try:
                self.plot_fit(wave, vtrig, left, right, par, chi2ndf0, ax=ax, show=show)
            except Exception:
                # Keep fitting robust even if plotting environment is unavailable
                pass

        return par, chi2ndf0

    # --- Convenience plotting helpers ----------------------------------------------------
    def _synthesize(self, x: np.ndarray, par: np.ndarray) -> np.ndarray:
        """Synthesize model values on grid `x` without saturation rejection."""
        return self._model_samples_vec(x, par)
