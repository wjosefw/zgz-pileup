#!/usr/bin/env bash
# Run the whole analysis, step by step, with the settings of config.py.
# Extra arguments go to every script, e.g.  ./run_all.sh --channels 1
set -e
cd "$(dirname "$0")"
for step in 01_read_data 02_clean_pulses 03_tail 04_tm_params 05_train_tm 06_calibrate 07_benchmark 08_recover; do
    python scripts/$step.py "$@"
done
