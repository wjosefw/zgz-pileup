"""Settings of the analysis. This is the only file to edit before running the scripts."""

# Runs to analyse: a name of your choice -> the folder with c<ch>.zip (or c<ch>.txt) and out.txt
RUNS = {
    "RUN1": "/data/ZGZ/RUN1",
}

CHANNELS = [0, 1, 2]

# Where all outputs go (it is created if needed)
WORK = "/data/ZGZ/zgz-work"

# True: the clean pulses of all runs are pooled, per channel, to learn one pulse shape, tail and
# set of templates per channel (more statistics; needs the same gain in every run, which
# 02_clean_pulses.py checks). False: every run learns its own.
POOL_RUNS = True

# Energy of the main peak (keV) for the amplitude -> keV calibration; None leaves energies empty
PEAK_KEV = 766

# Processes for the TM fits (one CPU core each) and the CPU temperature (°C) at which they pause
WORKERS = 3
MAX_TEMP = 85

# Synthetic pile-up pairs in the benchmark, and the random seed
N_BENCHMARK = 5000
SEED = 0
