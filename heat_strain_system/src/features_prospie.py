"""Feature engineering + real heat-strain labelling for PROSPIE.

Label: core temperature risk class, using standard occupational heat-strain
action limits (consistent with ACGIH TLV / ISO 7933 practice):
    Low       : Tre < 38.0 C
    Moderate  : 38.0 <= Tre < 38.5 C
    High      : Tre >= 38.5 C
This is REAL ground truth (rectal thermistor), not a proxy.

Forecast target: the class at t + HORIZON_MIN minutes ahead, predicted from
a trailing window of sensor features ending at t. Core temperature and
Δcore are EXCLUDED from the input features (they define the label -- using
them as inputs would be trivial leakage). Sweat production is also
excluded: it is a single whole-trial total (post-trial body-mass loss),
not a real-time signal, so including it would leak end-of-trial
information backwards in time.

Sensor modalities used as input:
    cardio_resp : HR, breathing rate                        (chest strap - wearable)
    skin_thermal: 11-site skin temperature + insulated skin temp (wearable patches)
    environment : chamber/microclimate Ta/RH/wind, condition WBGT, solar,
                  work/rest phase, treadmill workload, elapsed exposure time
    individual  : age, sex, body mass, height, body fat %, VO2peak,
                  acclimation/training status, clothing type (static per person)
"""
from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore", category=Warning)
import numpy as np
import pandas as pd

HORIZON_MIN = 15
ROLL_MIN = 5
LABELS = ["Low", "Moderate", "High"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}

CARDIO_RESP = ["hr", "breathing_rate"]
SKIN_THERMAL = ["skin_abdomen", "skin_calf", "skin_chest", "skin_foot", "skin_hand",
                "skin_head", "skin_lowerarm", "skin_lowerback", "skin_thigh",
                "skin_upperarm", "skin_upperback", "skin_insulated"]
ENVIRONMENT = ["ta_chamber", "rh_chamber", "wind_chamber", "t_front_cloth", "rh_front_cloth",
               "cond_wbgt", "cond_ta", "cond_rh", "cond_abs_humidity", "solar_present",
               "solar_rad_1", "workrest", "treadmill_speed", "treadmill_grade", "t_min"]
INDIVIDUAL = ["age", "sex", "bodymass_kg", "height_cm", "bodyfat_pct", "vo2peak",
              "acclimation", "trainingstatus", "clothing"]

MODALITIES = {
    "cardio_resp": CARDIO_RESP,
    "skin_thermal": SKIN_THERMAL,
    "environment": ENVIRONMENT,
    "individual": INDIVIDUAL,
}
ALL_BASE_COLS = CARDIO_RESP + SKIN_THERMAL + ENVIRONMENT + INDIVIDUAL


def core_to_class(t):
    if pd.isna(t):
        return None
    if t < 38.0:
        return "Low"
    if t < 38.5:
        return "Moderate"
    return "High"


def build_trial_features(trial: pd.DataFrame) -> pd.DataFrame:
    trial = trial.sort_values("t_min").reset_index(drop=True)
    rows = []
    times = trial["t_min"].values
    t_max = times.max()

    roll_cols = CARDIO_RESP + SKIN_THERMAL + ["ta_chamber", "rh_chamber", "wind_chamber",
                                               "t_front_cloth", "rh_front_cloth"]
    for col in roll_cols:
        trial.loc[:, f"{col}_roll_mean"] = trial[col].rolling(ROLL_MIN, min_periods=2).mean()
        trial.loc[:, f"{col}_roll_slope"] = trial[col].diff(ROLL_MIN - 1) / max(ROLL_MIN - 1, 1)
    trial = trial.copy()

    trial["y_now"] = trial["core_temp"].apply(core_to_class)

    for i in range(len(trial)):
        t0 = times[i]
        if t0 < ROLL_MIN:  # need enough history for rolling stats
            continue
        t_target = t0 + HORIZON_MIN
        if t_target > t_max:
            continue
        # nearest row to the target time (data is ~1-min spaced but not perfectly)
        j = int(np.argmin(np.abs(times - t_target)))
        if abs(times[j] - t_target) > 3:
            continue
        target_class = trial["y_now"].iloc[j]
        if target_class is None:
            continue

        row = {"participant": trial["participant"].iloc[i], "condition": trial["condition"].iloc[i],
               "scenario": trial["scenario"].iloc[i], "t_min": t0, "event": trial["event"].iloc[i]}
        for col in ALL_BASE_COLS:
            row[col] = trial[col].iloc[i]
        for col in roll_cols:
            row[f"{col}_roll_mean"] = trial[f"{col}_roll_mean"].iloc[i]
            row[f"{col}_roll_slope"] = trial[f"{col}_roll_slope"].iloc[i]
        row["core_temp_now"] = trial["core_temp"].iloc[i]  # kept only for reporting, not a feature
        row["label_now"] = trial["y_now"].iloc[i]
        row["label_next"] = target_class
        rows.append(row)
    return pd.DataFrame(rows)


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (p, c), trial in df.groupby(["participant", "condition"]):
        if len(trial) < ROLL_MIN + HORIZON_MIN:
            continue
        f = build_trial_features(trial)
        if len(f):
            frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out["y"] = out["label_next"].map(LABEL2IDX)
    return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from load_prospie import load_prospie
    df = load_prospie()
    feat = build_dataset(df)
    print(feat.shape, feat["participant"].nunique(), "participants")
    print(feat["y"].value_counts().rename(index=dict(enumerate(LABELS))))
    os.makedirs(os.path.join(os.environ.get("HS_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/tables"), exist_ok=True)
    feat.to_parquet(os.path.join(os.environ.get("HS_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/tables/prospie_features.parquet"))
    print("saved.")
