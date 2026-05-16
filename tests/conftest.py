"""
Спільні pytest-fixtures.
Всі тести використовують синтетичні дані.
"""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
def _make_online_rows(n: int, lat0: float = 48.46, lon0: float = 35.05,
                      behavior: str = "normal") -> str:
    """Генерує n рядків SEMANTIC_ONLINE.txt."""
    rng = np.random.default_rng(42)
    lines = []
    ratios = {"normal": (0.8, 0.1, 0.1),
              "drowsy": (0.1, 0.8, 0.1),
              "aggressive": (0.1, 0.1, 0.8)}
    rn, rd, ra = ratios.get(behavior, (0.8, 0.1, 0.1))
    for i in range(n):
        ts = float(i * 5)
        lat = lat0 + rng.normal(0, 0.001)
        lon = lon0 + rng.normal(0, 0.001)
        scores = [rng.uniform(60, 100) for _ in range(8)]
        row = [ts, lat, lon, *scores, rn, rd, ra, 0,
               *scores, rn, rd, ra, 0]
        lines.append(" ".join(f"{v:.6f}" for v in row))
    return "\n".join(lines)


def _make_final_values() -> str:
    vals = [8, 30, 0,            # start h/m/s
            65.5, 120.0, 85.0,   # avg/max speed, lanex
            10.5,                # driving_time_min
            8, 40, 30,           # end h/m/s
            12.3]                # distance_km
    vals += [80.0] * 43
    return "\n".join(str(v) for v in vals[:54])


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> Path:
    """ Створює мінімальний синтетичний датасет """
    trips = [
        ("D1", "20151110175712-12km-D1-NORMAL1-SECONDARY",    "normal",     48.46, 35.05),
        ("D1", "20151110180824-8km-D1-DROWSY-SECONDARY",      "drowsy",     48.47, 35.06),
        ("D2", "20151111123124-15km-D2-AGGRESSIVE-MOTORWAY",  "aggressive", 48.48, 35.07),
        ("D2", "20151111125233-10km-D2-NORMAL-MOTORWAY",      "normal",     48.49, 35.08),
    ]
    for driver, folder_name, behavior, lat0, lon0 in trips:
        folder = tmp_path / driver / folder_name
        folder.mkdir(parents=True)
        (folder / "SEMANTIC_ONLINE.txt").write_text(
            _make_online_rows(60, lat0, lon0, behavior)
        )
        (folder / "SEMANTIC_FINAL.txt").write_text(_make_final_values())

    return tmp_path


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """ DataFrame, що імітує вивід UAHDataLoader.load_full_dataset() """
    rng = np.random.default_rng(0)
    n = 240
    behaviors = (["normal"] * 60 + ["drowsy"] * 60 +
                 ["aggressive"] * 60 + ["normal"] * 60)

    df = pd.DataFrame({
        "timestamp_sec":           np.tile(np.arange(60) * 5, 4).astype(float),
        "gps_latitude":            48.46 + rng.normal(0, 0.005, n),
        "gps_longitude":           35.05 + rng.normal(0, 0.005, n),
        "score_total_window":      rng.uniform(50, 100, n),
        "score_acc_window":        rng.uniform(50, 100, n),
        "score_brake_window":      rng.uniform(50, 100, n),
        "score_turn_window":       rng.uniform(50, 100, n),
        "score_weave_window":      rng.uniform(50, 100, n),
        "score_drift_window":      rng.uniform(50, 100, n),
        "score_overspeed_window":  rng.uniform(50, 100, n),
        "score_carfollow_window":  rng.uniform(50, 100, n),
        "ratio_normal_window":     [0.8 if b == "normal" else 0.1 for b in behaviors],
        "ratio_drowsy_window":     [0.8 if b == "drowsy" else 0.1 for b in behaviors],
        "ratio_aggressive_window": [0.8 if b == "aggressive" else 0.1 for b in behaviors],
        "ratio_distracted_window": rng.integers(0, 2, n).astype(float),
        "score_total":             rng.uniform(50, 100, n),
        "score_acc":               rng.uniform(50, 100, n),
        "score_brake":             rng.uniform(50, 100, n),
        "score_turn":              rng.uniform(50, 100, n),
        "score_weave":             rng.uniform(50, 100, n),
        "score_drift":             rng.uniform(50, 100, n),
        "score_overspeed":         rng.uniform(50, 100, n),
        "score_carfollow":         rng.uniform(50, 100, n),
        "ratio_normal":            [0.8 if b == "normal" else 0.1 for b in behaviors],
        "ratio_drowsy":            [0.8 if b == "drowsy" else 0.1 for b in behaviors],
        "ratio_aggressive":        [0.8 if b == "aggressive" else 0.1 for b in behaviors],
        "ratio_distracted":        rng.integers(0, 2, n).astype(float),
        "trip_id":                 (["D1_trip1"] * 60 + ["D1_trip2"] * 60 +
                                    ["D2_trip1"] * 60 + ["D2_trip2"] * 60),
        "driver_pseudonym":        (["DRV_AABBCCDD1122"] * 120 +
                                    ["DRV_EEFF00112233"] * 120),
        "behavior_label":          behaviors,
        "road_type":               (["motorway"] * 120 + ["secondary"] * 120),
    })
    return df
