"""
Анонімізація транспортних даних.

Реалізує дворівневу анонімізацію:
    1. k-анонімність  – кожен запис невідрізний від ≥ (k-1) інших
                        за квазі-ідентифікаторами.
    2. l-різноманітність – кожен клас еквівалентності містить
                           ≥ l різних значень чутливого атрибуту.

Квазі-ідентифікатори (QI):
    • qi_geohash     – просторова комірка GPS (алгоритм Geohash)
    • qi_time_bucket – дискретний часовий інтервал (за замовч. 5 хв)
    • qi_speed_cat   – категорія швидкості (low / medium / high)

Чутливий атрибут (SA):
    • sa_behavior    – стиль водіння (normal / drowsy / aggressive)

Прямі ідентифікатори, що видаляються:
    gps_latitude, gps_longitude, timestamp_sec, trip_id
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    GEOHASH_PRECISION,
    GEOHASH_PRECISION_KM,
    SPEED_LABELS,
    SPEED_THRESHOLDS,
    TIME_BUCKET_SECONDS,
)

logger = logging.getLogger(__name__)

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"

def geohash_encode(lat: float, lon: float, precision: int = 5) -> str:
    """
    Кодує (lat, lon) у рядок Geohash заданої точності.

    Точність → приблизний розмір комірки:
        5 → ~2.4 km   (рекомендовано для k-анонімності)
        4 → ~20 km
        3 → ~78 km
    """
    if pd.isna(lat) or pd.isna(lon):
        return "unknown"
    lat = float(lat)
    lon = float(lon)

    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    result = []
    bits = [16, 8, 4, 2, 1]
    bit_idx = 0
    even = True
    ch = 0

    while len(result) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2
            if lon >= mid:
                ch |= bits[bit_idx]
                lon_lo = mid
            else:
                lon_hi = mid
        else:
            mid = (lat_lo + lat_hi) / 2
            if lat >= mid:
                ch |= bits[bit_idx]
                lat_lo = mid
            else:
                lat_hi = mid
        even = not even
        if bit_idx < 4:
            bit_idx += 1
        else:
            result.append(_BASE32[ch])
            bit_idx = 0
            ch = 0

    return "".join(result)


def geohash_decode_center(code: str) -> Tuple[float, float]:
    """Повертає (lat, lon) центру комірки геохешу."""
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    even = True

    for char in code:
        cd = _BASE32.index(char)
        for mask in [16, 8, 4, 2, 1]:
            if even:
                mid = (lon_lo + lon_hi) / 2
                lon_lo, lon_hi = (mid, lon_hi) if cd & mask else (lon_lo, mid)
            else:
                mid = (lat_lo + lat_hi) / 2
                lat_lo, lat_hi = (mid, lat_hi) if cd & mask else (lat_lo, mid)
            even = not even

    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2

@dataclass
class AnonymizationReport:
    """Структурований звіт про результати анонімізації."""

    k_parameter: int = 3
    l_parameter: int = 2
    geohash_precision_initial: int = 5
    time_bucket_seconds: int = 300

    original_records: int = 0
    after_k_anonymity: int = 0
    after_l_diversity: int = 0
    anonymized_records: int = 0
    suppressed_records: int = 0
    suppression_rate: float = 0.0

    num_equivalence_classes: int = 0
    avg_class_size: float = 0.0
    min_class_size: int = 0
    max_class_size: int = 0
    applied_geohash_precision: int = 5
    applied_precision_km: float = 2.4

    k_iterations: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  ЗВІТ АНОНІМІЗАЦІЇ ТРАНСПОРТНИХ ДАНИХ",
            "=" * 55,
            f"  Параметри:     k={self.k_parameter}, l={self.l_parameter}",
            f"  Геохеш:        precision={self.applied_geohash_precision} "
            f"(~{self.applied_precision_km:.1f} km)",
            f"  Часовий bucket:{self.time_bucket_seconds} с",
            "-" * 55,
            f"  Вхід:          {self.original_records:>8} записів",
            f"  Після k-ан.:   {self.after_k_anonymity:>8} записів",
            f"  Після l-різн.: {self.after_l_diversity:>8} записів",
            f"  Пригнічено:    {self.suppressed_records:>8} записів "
            f"({self.suppression_rate:.1%})",
            "-" * 55,
            f"  Класів еквів.: {self.num_equivalence_classes}",
            f"  Розмір класу:  avg={self.avg_class_size:.1f}, "
            f"min={self.min_class_size}, max={self.max_class_size}",
            "=" * 55,
        ]
        return "\n".join(lines)

_QI_COLS = ["qi_geohash", "qi_time_bucket", "qi_speed_cat"]
_DIRECT_ID_COLS = ["gps_latitude", "gps_longitude", "timestamp_sec", "trip_id"]

class TransportDataAnonymizer:
    """ Анонімізатор транспортних даних."""

    def __init__(
        self,
        k: int = 3,
        l: int = 2,
        geohash_precision: int = GEOHASH_PRECISION,
        time_bucket_sec: int = TIME_BUCKET_SECONDS,
    ) -> None:
        if k < 2:
            raise ValueError("k має бути ≥ 2")
        if l < 2:
            raise ValueError("l має бути ≥ 2")
        self.k = k
        self.l = l
        self.geohash_precision = geohash_precision
        self.time_bucket_sec = time_bucket_sec

    def _speed_category(self, val) -> str:
        """Категоризує числове значення у low / medium / high."""
        if pd.isna(val) or val < 0:
            return "unknown"
        for i, thr in enumerate(SPEED_THRESHOLDS[1:]):
            if val < thr:
                return SPEED_LABELS[i]
        return SPEED_LABELS[-1]

    def _dominant_behavior(self, row: pd.Series) -> str:
        """
        Визначає домінуючу поведінку водія.
        Пріоритет: явна мітка behavior_label > window ratios > cumulative ratios.
        """
        if "behavior_label" in row and pd.notna(row["behavior_label"]):
            return str(row["behavior_label"]).strip().lower()

        candidates = {}
        for key in ("ratio_normal_window", "ratio_drowsy_window", "ratio_aggressive_window"):
            v = row.get(key)
            if pd.notna(v):
                behavior = key.replace("ratio_", "").replace("_window", "")
                candidates[behavior] = float(v)
        if not candidates:
            for key in ("ratio_normal", "ratio_drowsy", "ratio_aggressive"):
                v = row.get(key)
                if pd.notna(v):
                    behavior = key.replace("ratio_", "")
                    candidates[behavior] = float(v)

        return max(candidates, key=candidates.get) if candidates else "unknown"

    def compute_quasi_identifiers(
        self, df: pd.DataFrame, geohash_precision: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Обчислює квазі-ідентифікатори та чутливий атрибут.

        Додає стовпці: qi_geohash, qi_time_bucket, qi_speed_cat, sa_behavior.
        Оригінальні координати та час ще не видаляються на цьому кроці.
        """
        precision = geohash_precision or self.geohash_precision
        result = df.copy()

        result["qi_geohash"] = result.apply(
            lambda r: geohash_encode(
                r.get("gps_latitude", np.nan),
                r.get("gps_longitude", np.nan),
                precision,
            ),
            axis=1,
        )

        result["qi_time_bucket"] = (
            result["timestamp_sec"]
            .apply(lambda t: int(t // self.time_bucket_sec) if pd.notna(t) else -1)
        )

        if "speed_kmh" in result.columns:
            result["qi_speed_cat"] = result["speed_kmh"].apply(self._speed_category)
        else:
            result["qi_speed_cat"] = result["score_overspeed_window"].apply(
                lambda s: "low" if pd.isna(s) or s >= 80
                else ("medium" if s >= 50 else "high")
            )

        result["sa_behavior"] = result.apply(self._dominant_behavior, axis=1)

        return result

    def k_anonymize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Застосовує k-анонімність із адаптивним зниженням точності геохешу.

        Алгоритм:
        1. Обчислюємо розмір кожного класу еквівалентності.
        2. Якщо >20% записів не задовольняють k → знижуємо точність геохешу.
        3. Продовжуємо до precision=1 або задовільного рівня пригнічення.
        4. Записи у класах менших за k — видаляються (suppression).
        """
        current_precision = self.geohash_precision
        result = df.copy()
        iterations = []

        while current_precision >= 1:
            if current_precision != self.geohash_precision:
                result["qi_geohash"] = result.apply(
                    lambda r: geohash_encode(
                        r.get("gps_latitude", np.nan),
                        r.get("gps_longitude", np.nan),
                        current_precision,
                    ),
                    axis=1,
                )

            class_sizes = result.groupby(_QI_COLS)["sa_behavior"].transform("count")
            k_mask = class_sizes >= self.k
            satisfied = k_mask.mean()

            iter_info = {
                "precision": current_precision,
                "records_total": len(result),
                "records_satisfy_k": int(k_mask.sum()),
                "suppression_rate": float(1 - satisfied),
            }
            iterations.append(iter_info)
            logger.debug(
                "  k-ан. precision=%d: %.1f%% задовольняють k=%d",
                current_precision, satisfied * 100, self.k,
            )

            if satisfied >= 0.80 or current_precision == 1:
                result = result[k_mask].copy()
                result["applied_geohash_precision"] = current_precision
                break

            current_precision -= 1

        return result, iterations

    def l_diversify(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Застосовує l-різноманітність.

        Видаляє цілі класи еквівалентності, де кількість унікальних
        значень SA < l. Це запобігає атакам на однорідні групи.
        """
        diversity = df.groupby(_QI_COLS)["sa_behavior"].transform("nunique")
        l_mask = diversity >= self.l

        stats = {
            "total_classes": df.groupby(_QI_COLS).ngroups,
            "satisfying_classes": int(
                df[l_mask].groupby(_QI_COLS).ngroups if l_mask.any() else 0
            ),
            "suppressed_records": int((~l_mask).sum()),
            "kept_records": int(l_mask.sum()),
        }
        logger.debug(
            "  l-різн.: %d/%d класів задовольняють l=%d",
            stats["satisfying_classes"], stats["total_classes"], self.l,
        )
        return df[l_mask].copy(), stats

    def anonymize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, AnonymizationReport]:
        """
        Повний пайплайн анонімізації:

        1. Обчислення QI (geohash, time_bucket, speed_cat, behavior)
        2. k-анонімність (з адаптивним geohash precision)
        3. l-різноманітність
        4. Видалення прямих ідентифікаторів

        Returns:
            df_anon  – анонімізований DataFrame
            report   – деталізований звіт
        """
        report = AnonymizationReport(
            k_parameter=self.k,
            l_parameter=self.l,
            geohash_precision_initial=self.geohash_precision,
            time_bucket_seconds=self.time_bucket_sec,
            original_records=len(df),
        )

        logger.info("Крок 1: Обчислення квазі-ідентифікаторів...")
        df_qi = self.compute_quasi_identifiers(df)

        logger.info("Крок 2: k-анонімність (k=%d)...", self.k)
        df_k, k_iters = self.k_anonymize(df_qi)
        report.after_k_anonymity = len(df_k)
        report.k_iterations = k_iters
        applied_precision = int(df_k["applied_geohash_precision"].iloc[0]) if not df_k.empty else 1
        report.applied_geohash_precision = applied_precision
        report.applied_precision_km = GEOHASH_PRECISION_KM.get(applied_precision, 0)

        logger.info("Крок 3: l-різноманітність (l=%d)...", self.l)
        df_l, l_stats = self.l_diversify(df_k)
        report.after_l_diversity = len(df_l)

        logger.info("Крок 4: Видалення прямих ідентифікаторів...")
        cols_to_drop = [c for c in _DIRECT_ID_COLS if c in df_l.columns]
        df_anon = df_l.drop(columns=cols_to_drop)

        report.anonymized_records = len(df_anon)
        report.suppressed_records = report.original_records - report.anonymized_records
        report.suppression_rate = report.suppressed_records / max(report.original_records, 1)

        if not df_anon.empty:
            class_sizes = df_anon.groupby(_QI_COLS).size()
            report.num_equivalence_classes = len(class_sizes)
            report.avg_class_size = float(class_sizes.mean())
            report.min_class_size = int(class_sizes.min())
            report.max_class_size = int(class_sizes.max())

        logger.info(report.summary())
        return df_anon, report
