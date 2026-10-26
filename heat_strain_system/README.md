# Leakage-safe heat-strain forecasting on PROSPIE (research code)

Forecasts rectal-temperature risk class (Low <38.0, Moderate 38.0-38.5, High >=38.5 C) 15 minutes ahead from
wearable-style signals, validated by participant-grouped cross-validation. Research prototype: NOT validated for
real-world safety use.

## Run
1. `pip install -r requirements.txt`
2. Download the open-access PROSPIE workbook (`Prospie_dataset_open_access_V3.xlsx`) and place it in `data/` (or set `HS_DATA`).
3. `bash run_all.sh` (neural-network steps are pure NumPy and take a while on CPU).

Results land in `results/` (tables, `results/extra/` for out-of-fold predictions, metrics and figures).
`results/` in this package already holds the outputs reported in the paper, so figures can be re-drawn without retraining:
`cd src && HS_ROOT=.. python make_extra_outputs.py`.

## Layout (src/)
- `load_prospie.py`, `features_prospie.py`: reading, 5-min window features, labels. Tre, change in core, and whole-trial sweat loss are never inputs.
- `nn_engine.py`: small NumPy autodiff engine (gradient-checked). `fusion_model_prospie.py`: attention-gated fusion network.
- `baselines.py`, `common.py`: LogReg / RF / XGBoost, metrics, imputation (train fold only), calibration.
- `evaluate_prospie.py`, `loco_lopo_analysis.py`: main CV, LOPO, LOCO.
- `oof_fusion.py`, `oof_classical.py`, `make_extra_outputs.py`, `missing_subset.py`, `event_level.py`: pooled out-of-fold analyses, confusion matrices, trajectories, fusion weights, sanity checks.

## Honest notes
- Fusion results are not bit-for-bit reproducible across library versions; aggregates agree within ~0.007 macro-F1.
- Attention weights vary across seeds; do not read them as sensor importance.
- The first calibration protocol was wrong (temperature fitted on training participants); review_calibration.py is the corrected one.
- High false-alarm rates at the default decision rule (see paper, Section XI-B).
