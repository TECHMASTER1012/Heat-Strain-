"""Loader for the PROSPIE open-access heat-strain dataset (per-minute,
40 participants, 9 real heat-stress conditions, real rectal core
temperature ground truth). See the Excel's 'Key' and 'Conditions' sheets
for the original column documentation this is transcribed from."""
from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd

XLSX_PATH = os.environ.get("HS_DATA", "data/Prospie_dataset_open_access_V3.xlsx")

COLS = [
    "condition", "time", "event",                       # 0-2
    "participant", "sex", "age", "bodymass_kg", "height_cm", "bodyfat_pct",
    "vo2peak", "acclimation", "trainingstatus", "clothing",  # 3-12
    "rh_chamber", "ta_chamber", "wind_chamber",           # 13-15
    "t_front_cloth", "rh_front_cloth", "t_back_cloth", "rh_back_cloth",  # 16-19
    "workrest", "vo2_work", "treadmill_speed", "treadmill_grade",  # 20-23
    "solar_present", "solar_rad_1", "solar_rad_2", "solar_rad_3",  # 24-27
    "sweat_l",                                            # 28
    "core_temp", "d_core",                                # 29-30
    "skin_abdomen", "skin_calf", "skin_chest", "skin_foot", "skin_hand",
    "skin_head", "skin_lowerarm", "skin_lowerback", "skin_thigh",
    "skin_upperarm", "skin_upperback",                    # 31-41
    "skin_insulated", "t_amb_neck",                       # 42-43
    "breathing_rate", "hr", "hr_type",                    # 44-46
    "ts_sensation", "tc_comfort", "tp_preference", "sw_wetness", "rpe",  # 47-51
    "rr_manual", "sdrr_manual", "rmssd_manual",            # 52-54
    "rr_software", "sdrr_software", "rmssd_software",      # 55-57
]

MISSING_SENTINEL = 9999


def parse_condition_table():
    """Extract numeric Ta/RH/WBGT/abs.humidity/clothing/solar from the free
    -text 'Conditions' sheet and attach the scenario description."""
    cond = pd.read_excel(XLSX_PATH, sheet_name="Conditions ", header=None, skiprows=2)
    cond.columns = ["condition", "ambient_text", "work_pattern", "scenario"]
    cond = cond.dropna(subset=["condition"])
    cond["condition"] = cond["condition"].astype(int)

    def parse_ambient(s):
        # e.g. "40°C, 35 %, 31.85°C, 17.90g.m-3, Permeable, no solar"
        nums = re.findall(r"[-+]?\d*\.?\d+", str(s))
        ta = float(nums[0]) if len(nums) > 0 else np.nan
        rh = float(nums[1]) if len(nums) > 1 else np.nan
        wbgt = float(nums[2]) if len(nums) > 2 else np.nan
        abs_hum = float(nums[3]) if len(nums) > 3 else np.nan
        permeable = 1 if "permeable" in str(s).lower() and "impermeable" not in str(s).lower() else 0
        solar = 1 if "no solar" not in str(s).lower() else 0
        return pd.Series([ta, rh, wbgt, abs_hum, permeable, solar])

    cond[["cond_ta", "cond_rh", "cond_wbgt", "cond_abs_humidity", "cond_permeable", "cond_solar"]] = \
        cond["ambient_text"].apply(parse_ambient)
    return cond[["condition", "cond_ta", "cond_rh", "cond_wbgt", "cond_abs_humidity",
                 "cond_permeable", "cond_solar", "scenario"]]


def load_prospie() -> pd.DataFrame:
    df = pd.read_excel(XLSX_PATH, sheet_name="Data", header=None, skiprows=2)
    df.columns = COLS[: df.shape[1]]
    df = df.replace(MISSING_SENTINEL, np.nan)

    # Time -> minutes since trial start (per participant+condition)
    def to_minutes(t):
        try:
            return t.hour * 60 + t.minute + t.second / 60.0
        except AttributeError:
            return np.nan
    df["t_min"] = df["time"].apply(to_minutes)

    cond_table = parse_condition_table()
    df = df.merge(cond_table, on="condition", how="left")

    df = df.dropna(subset=["core_temp"]).reset_index(drop=True)
    df = df.sort_values(["participant", "condition", "t_min"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_prospie()
    print(df.shape)
    print(df[["participant", "condition", "t_min", "core_temp", "hr", "cond_wbgt", "scenario"]].head(10))
    print("\nmissingness:")
    print(df.isna().mean().sort_values(ascending=False).head(20))
