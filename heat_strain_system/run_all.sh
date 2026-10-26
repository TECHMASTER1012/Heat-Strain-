#!/usr/bin/env bash
# Usage: HS_DATA=/path/to/Prospie_dataset_open_access_V3.xlsx bash run_all.sh
set -e
cd "$(dirname "$0")/src"
export HS_ROOT="$(cd .. && pwd)"
export HS_DATA="${HS_DATA:-$HS_ROOT/data/Prospie_dataset_open_access_V3.xlsx}"
python features_prospie.py          # 1. load, window, label, build 76 features
python evaluate_prospie.py          # 2. 8-fold grouped CV: baselines + fusion + ablations + calibration
python loco_lopo_analysis.py        # 3. leave-one-participant / leave-one-scenario-out
for s in 0 1 2; do HS_SEED=$s python oof_fusion.py; done   # 4. out-of-fold fusion predictions, 3 seeds
python oof_classical.py             # 5. out-of-fold LogReg/RF/XGBoost, 3 seeds
python missing_subset.py            # 6. missingness + complete HR/BR rerun
python event_level.py               # 7. lead time / false alarms
python make_plots_prospie.py        # 8. original figures
python make_extra_outputs.py        # 9. confusion, ROC/PR, trajectories, attention, permutation test, tables
# --- additions for the review revision ---
python review_baselines.py          # 10. reference predictors, Holm tests, bootstrap, realistic sensor sets
python review_calibration.py        # 11. corrected calibration protocol + MC-dropout abstention
python review_event.py              # 12. event-level lead/false alarms (explicit definition)
python review_partitions.py         # 13. five random participant partitions
for s in 0; do HS_SEED=$s python review_fusion_modalities.py; done   # 14. fusion single / leave-one-modality-out
python review_figs.py               # 15. redraw figures
