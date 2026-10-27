"""Optional. Write clean pulses as fixed-length float32 records, completed with the measured tail,
for tools that need every pulse in full (template training elsewhere, synthetic pile-up,
neural-network training).

Each pulse is placed with its 50 % rise at sample EXPORT_PRE + t50_median, preceded by baseline
noise; after the end of its record it is continued with its amplitude times the TM unit pulse
(tail and undershoot) plus baseline noise, up to the length of the longest record. Samples are
rounded to integer ADC counts like the recorded ones.

  runs/<run>/ch<c>/export/clean_pulses_<L>.dat, clean_pulses_info.npz, README.txt

    python scripts/export_clean_pulses.py [--runs ...] [--channels ...] [--n 100000]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zgzpileup import io, tm  # noqa: E402
from zgzpileup.project import banner, load_npz, need, parse, summary, update  # noqa: E402

EXPORT_PRE = 64        # extra baseline samples before each pulse

proj, args = parse(__doc__.split("\n")[0], lambda p: p.add_argument(
    "--n", type=int, default=100_000, help="number of clean pulses written per run and channel"))
for run in proj.runs:
    for ch in proj.channels:
        banner(f"{run} channel {ch}: export clean pulses")
        d = proj.run_dir(run, ch)
        md = proj.model_dir(proj.group_of(run), ch)
        need(d, "records.npz", "pulses.npz")
        need(md, "tm_model.npz")
        rec = io.Records.load(os.path.join(d, "records.npz"))
        P = load_npz(os.path.join(d, "pulses.npz"))
        model = tm.Model.load(os.path.join(md, "tm_model.npz"))
        s = model.s
        cls, base, amp, t50 = P["cls"], P["baseline"].astype(float), P["amp"], P["t50"]
        noise = summary(md).get("shape", {}).get("baseline_noise_adc", 0.6)
        ref, D, K1 = int(s["ref"]), int(s["D"]), int(s["K1"])
        L = int(np.ceil((EXPORT_PRE + ref + D + K1 + 1) / 64) * 64)
        rng = np.random.default_rng(proj.seed + 2)
        idx = np.nonzero(cls == 0)[0]
        sel = np.sort(rng.choice(idx, min(args.n, len(idx)), replace=False))
        out_dir = os.path.join(d, "export"); os.makedirs(out_dir, exist_ok=True)
        name = f"clean_pulses_{L}.dat"
        mm = np.memmap(os.path.join(out_dir, name), dtype="<f4", mode="w+", shape=(len(sel), L))
        k_unit, y_unit = model.unit_pulse()
        for c0 in range(0, len(sel), 20_000):
            part = sel[c0:c0 + 20_000]
            block = rng.normal(0, noise, (len(part), L))
            for r, i in enumerate(part):
                x = rec.wave(i, base[i])
                start = EXPORT_PRE + ref - t50[i]
                n = min(len(x), L - start)
                block[r, start:start + n] = x[:n]
                k = np.arange(start + n, L) - (start + t50[i] + D)
                block[r, start + n:] += amp[i] * np.interp(k, k_unit, y_unit, left=0, right=y_unit[-1])
            mm[c0:c0 + len(part)] = np.round(block)
        mm.flush(); del mm
        np.savez(os.path.join(out_dir, "clean_pulses_info.npz"), record=sel, amplitude_adc=amp[sel],
                 t50_sample=EXPORT_PRE + ref, recorded_samples=rec.length[sel])
        info = (f"clean_pulses_{L}.dat: {len(sel):,} clean pulses of {run} channel {ch}, float32 little-endian, "
                f"no header, {L} samples each (record after record), sampling {rec.dt_us * 1e3:.0f} ns, ADC counts, "
                f"baseline 0.\n"
                f"Each pulse has its 50 % rise at sample {EXPORT_PRE + ref}. The recorded samples are kept as they are; "
                f"before them there is baseline noise ({noise:.2f} ADC rms), and after the end of the record the pulse is "
                f"continued with its amplitude times the TM unit pulse (measured tail and undershoot) plus noise.\n"
                f"clean_pulses_info.npz: record number, amplitude (ADC), recorded length of every pulse.\n"
                f"Load with numpy: np.fromfile('{name}', dtype='<f4').reshape(-1, {L})\n")
        open(os.path.join(out_dir, "README.txt"), "w").write(info)
        print(info)
        update(d, "export", dict(file=name, pulses=int(len(sel)), samples=L, t50_sample=EXPORT_PRE + ref))
