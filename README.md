# Forecasting occupational heat-strain risk 15 minutes ahead

BCSE316L Course Project, School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology.

| Team member | Register no. |
|---|---|
| Saksham Mishra | 23BCT0228 |
| Shashwat Trivedi | 23BCT0175 |
| Raunak Saigal | 23BCT0193 |

Supervisor: Prof. Prabin S M (prabin.sm@vit.ac.in)

## What this project does

Workers in hot environments need a warning before their core temperature becomes dangerous, but wearable sensors cannot measure core temperature directly. This code forecasts a worker's heat-strain risk class 15 minutes ahead from heart rate, breathing rate, skin temperature, environmental readings and a few personal descriptors.

The risk class comes from measured rectal temperature:

| Class | Rectal temperature |
|---|---|
| Low | below 38.0 °C |
| Moderate | 38.0 to 38.5 °C |
| High | 38.5 °C or above |

Rectal temperature, its change from baseline and whole-trial sweat loss are never used as inputs. Every reported score comes from participants or scenarios that were held out of training.

The data is the open PROSPIE dataset from Loughborough University (40 participants, 9 heat-stress scenarios). After windowing it gives 14,103 labelled five-minute windows with 76 features in four sensor groups: cardio-respiratory, skin thermal, environment and individual.

## Main results

Eight-fold cross-validation grouped by participant:

| Model | Macro-F1 | Accuracy |
|---|---|---|
| Always predict Low | 0.219 | 0.489 |
| WBGT threshold rule | 0.334 | 0.356 |
| PHS (ISO 7933), 38.0/38.5 thresholds | 0.467 | 0.538 |
| Logistic regression | 0.673 | 0.697 |
| Random forest | 0.692 | 0.708 |
| Attention-gated fusion network | 0.674 | 0.693 |
| **XGBoost** | **0.719** | **0.738** |

- Skin temperature carries most of the signal. A chest-patch sensor set (HR, BR, chest skin temperature, time) reaches a macro-F1 of 0.624.
- Leave-one-participant-out accuracy averages 0.736. Leave-one-scenario-out accuracy averages 0.682, and full protective clothing is the hardest scenario.
- With the corrected calibration protocol, temperature scaling lowers the fusion network's expected calibration error from 0.092 to 0.032.
- At the default threshold the models raise 2.8 to 5.0 false High forecasts per hour. This is a research prototype and must not be used as a safety alarm.

## Repository layout

```
heat_strain_system/
  src/                  all code (see below)
  results/tables/       main result tables and the cached feature table
  results/extra/        out-of-fold predictions, extra metrics, logs, figures
  requirements.txt
  run_all.sh            runs the full pipeline in order
  Heat_Strain_Extended_IEEE_Paper.pdf
```

Main modules in `src/`:

| File | What it does |
|---|---|
| `load_prospie.py` | Reads the PROSPIE workbook and parses the condition sheet into numbers |
| `features_prospie.py` | Builds the 5-minute windows, rolling features and the 15-minute-ahead label |
| `common.py` | Training-fold imputer, class weights, metrics, ECE, temperature scaling |
| `baselines.py` | Logistic regression, random forest and XGBoost |
| `nn_engine.py` | Small NumPy automatic-differentiation engine (gradient-checked) |
| `fusion_model_prospie.py` | Attention-gated fusion network with an optional gradient-reversal branch |
| `evaluate_prospie.py` | Grouped cross-validation, ablations and paired tests |
| `loco_lopo_analysis.py` | Leave-one-participant-out and leave-one-scenario-out runs |
| `review_*.py`, `event_level.py` | Reference predictors, corrected calibration, partitions, modality runs, warning timing |

PyTorch could not be installed in our environment, so the fusion network runs on `nn_engine.py` instead of a deep-learning framework.

## Running it

1. Install the dependencies (Python 3):

   ```bash
   cd heat_strain_system
   pip install -r requirements.txt
   pip install pythermalcomfort   # only needed for the PHS baseline (review_index_baselines.py)
   ```

2. Download the open-access PROSPIE workbook `Prospie_dataset_open_access_V3.xlsx` from the Loughborough University repository (https://doi.org/10.17028/rd.lboro.26076577) and put it in `heat_strain_system/data/`, or point `HS_DATA` at it.

3. Run the full pipeline:

   ```bash
   bash run_all.sh
   # or: HS_DATA=/path/to/Prospie_dataset_open_access_V3.xlsx bash run_all.sh
   ```

   Everything runs on a CPU. The neural-network steps are the slowest because they use plain NumPy.

The `results/` folder already holds the outputs used in the paper and report, so you can redraw the figures without retraining:

```bash
cd heat_strain_system/src
HS_ROOT=.. python make_extra_outputs.py
```

## Notes

- Fusion-network results are not bit-for-bit reproducible across library versions; aggregates agree within about 0.007 macro-F1.
- The attention weights change between random seeds, so they should not be read as sensor importance.
- The first calibration protocol was wrong: the temperature was fitted on participants the network had already trained on. `review_calibration.py` contains the corrected version, and only its results are reported.
- All data come from 40 people in a laboratory. Field use would need validation on real worksites.

## Data source

Loughborough University, "Dataset for 'Prediction of core body temperature from multiple variables' and 'The physiological strain index does not reliably identify individuals at risk of reaching a thermal tolerance limit' (PROSPIE project)," Loughborough University Research Repository, 2024, doi: [10.17028/rd.lboro.26076577](https://doi.org/10.17028/rd.lboro.26076577).
