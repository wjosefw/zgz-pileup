"""Step 6. Energy calibration: E = c * A, with c placing the photopeak of the clean pulses at
PEAK_KEV (config.py). Check: the sum peak (two photopeak pulses detected together) should then be
at about twice that energy. Without PEAK_KEV the energies are left empty.

  models/<group>/ch<c>/calibration.json

Figure: models/.../figures/07_energy_calibration.png

    python scripts/06_calibrate.py [--channels ...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zgzpileup import plots, pulses  # noqa: E402
from zgzpileup.project import banner, load_npz, need, parse, update  # noqa: E402

proj, _ = parse(__doc__.split("\n")[0])
peak_kev = getattr(proj.cfg, "PEAK_KEV", None)
for ch in proj.channels:
    for group, runs in proj.groups().items():
        banner(f"channel {ch}: energy calibration")
        md = proj.model_dir(group, ch)
        amp = []
        for r in runs:
            d = proj.run_dir(r, ch)
            need(d, "pulses.npz")
            P = load_npz(os.path.join(d, "pulses.npz"))
            amp.append(P["amp"][P["cls"] == 0])
        amp = np.concatenate(amp)
        a_peak, fwhm = pulses.photopeak(amp)
        a_sum = pulses.peak_between(amp, 1.7 * a_peak, 2.3 * a_peak)
        check = dict(photopeak_adc=a_peak, photopeak_fwhm_pct=100 * fwhm / a_peak, sum_peak_adc=a_sum,
                     sum_over_photopeak=a_sum / a_peak)
        if not peak_kev:
            print(f"no PEAK_KEV: energies will be left empty (photopeak at {a_peak:.2f} ADC)")
            json.dump(dict(slope=None, offset=None, method="none"), open(os.path.join(md, "calibration.json"), "w"), indent=1)
            update(md, "calibrate", dict(method="none", **check))
            continue
        slope = peak_kev / a_peak
        cal = dict(slope=float(slope), offset=0.0, method=f"proportional: photopeak at {a_peak:.2f} ADC set to {peak_kev} keV")
        json.dump(cal, open(os.path.join(md, "calibration.json"), "w"), indent=1)
        plots.calibration(md, (slope, 0.0), dict(amp=amp, kev=peak_kev, fwhm_pct=100 * fwhm / a_peak, sum_kev=slope * a_sum))
        print(f"{slope:.3f} keV per ADC; photopeak FWHM {100 * fwhm / a_peak:.1f} %; "
              f"sum peak at {slope * a_sum:.0f} keV ({a_sum / a_peak:.3f} × photopeak)")
        update(md, "calibrate", dict(cal, photopeak_kev=peak_kev, sum_peak_kev=slope * a_sum, **check))
