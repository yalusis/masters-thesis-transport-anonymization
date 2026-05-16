"""
Тести модуля src.anonymizer
Покриття: geohash, k-анонімність, l-різноманітність, повний пайплайн.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anonymizer import (
    TransportDataAnonymizer,
    AnonymizationReport,
    geohash_encode,
    geohash_decode_center,
)


class TestGeohash:

    def test_encode_returns_string(self):
        result = geohash_encode(48.46, 35.05, precision=5)
        assert isinstance(result, str)
        assert len(result) == 5

    def test_precision_controls_length(self):
        for p in [1, 2, 3, 4, 5, 6]:
            assert len(geohash_encode(48.46, 35.05, precision=p)) == p

    def test_known_coordinates(self):
        code = geohash_encode(48.46, 35.05, precision=5)
        assert len(code) == 5
        assert code != "unknown"

    def test_nearby_points_same_cell(self):
        base = geohash_encode(48.4600, 35.0500, precision=5)
        near = geohash_encode(48.4601, 35.0501, precision=5)
        assert base == near

    def test_distant_points_different_cells(self):
        kyiv  = geohash_encode(50.45, 30.52, precision=5)
        dnipro = geohash_encode(48.46, 35.05, precision=5)
        assert kyiv != dnipro

    def test_nan_coordinates_return_unknown(self):
        assert geohash_encode(float("nan"), 35.05) == "unknown"
        assert geohash_encode(48.46, float("nan")) == "unknown"

    def test_decode_roundtrip(self):
        lat, lon = 48.4600, 35.0500
        code = geohash_encode(lat, lon, precision=6)
        dec_lat, dec_lon = geohash_decode_center(code)
        assert abs(dec_lat - lat) < 0.05
        assert abs(dec_lon - lon) < 0.05

    def test_lower_precision_is_prefix(self):
        p5 = geohash_encode(48.46, 35.05, precision=5)
        p3 = geohash_encode(48.46, 35.05, precision=3)
        assert p5.startswith(p3)


class TestComputeQI:

    def test_qi_columns_added(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2)
        result = anon.compute_quasi_identifiers(raw_df)
        for col in ("qi_geohash", "qi_time_bucket", "qi_speed_cat", "sa_behavior"):
            assert col in result.columns

    def test_geohash_length(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2, geohash_precision=5)
        result = anon.compute_quasi_identifiers(raw_df)
        lengths = result["qi_geohash"].str.len()
        assert (lengths == 5).all()

    def test_time_bucket_is_integer(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2, time_bucket_sec=300)
        result = anon.compute_quasi_identifiers(raw_df)
        assert pd.api.types.is_integer_dtype(result["qi_time_bucket"])

    def test_time_bucket_range(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2, time_bucket_sec=300)
        result = anon.compute_quasi_identifiers(raw_df)
        assert (result["qi_time_bucket"] >= 0).all()

    def test_speed_category_values(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2)
        result = anon.compute_quasi_identifiers(raw_df)
        valid = {"low", "medium", "high", "unknown"}
        assert set(result["qi_speed_cat"].unique()).issubset(valid)

    def test_behavior_from_label(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2)
        result = anon.compute_quasi_identifiers(raw_df)
        expected = set(raw_df["behavior_label"].str.lower().unique())
        got = set(result["sa_behavior"].unique())
        assert expected == got

    def test_original_coords_preserved(self, raw_df):

        anon = TransportDataAnonymizer(k=3, l=2)
        result = anon.compute_quasi_identifiers(raw_df)
        assert "gps_latitude" in result.columns
        assert "gps_longitude" in result.columns

class TestKAnonymity:

    def _make_df_with_k_issue(self) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        rows = []
        for i in range(10):
            rows.append({
                "timestamp_sec": float(i * 5),
                "gps_latitude": 48.460 + rng.normal(0, 0.0001),
                "gps_longitude": 35.050 + rng.normal(0, 0.0001),
                "score_overspeed_window": 90.0,
                "behavior_label": "normal",
                "driver_pseudonym": "DRV_TEST",
                "trip_id": "T1",
            })
        rows.append({
            "timestamp_sec": 5000.0,
            "gps_latitude": 50.45,
            "gps_longitude": 30.52,
            "score_overspeed_window": 90.0,
            "behavior_label": "normal",
            "driver_pseudonym": "DRV_TEST",
            "trip_id": "T1",
        })
        return pd.DataFrame(rows)

    def test_k_anonymity_removes_small_classes(self):
        anon = TransportDataAnonymizer(k=3, l=2)
        df_raw = self._make_df_with_k_issue()
        df_qi = anon.compute_quasi_identifiers(df_raw)
        df_k, _ = anon.k_anonymize(df_qi)
        assert len(df_k) < len(df_qi)

    def test_all_classes_satisfy_k(self, raw_df):
        k = 3
        anon = TransportDataAnonymizer(k=k, l=2)
        df_qi = anon.compute_quasi_identifiers(raw_df)
        df_k, _ = anon.k_anonymize(df_qi)

        if not df_k.empty:
            sizes = df_k.groupby(["qi_geohash", "qi_time_bucket", "qi_speed_cat"]).size()
            assert (sizes >= k).all(), f"Знайдено класи менші за k={k}: {sizes[sizes < k]}"

    def test_iterations_logged(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2)
        df_qi = anon.compute_quasi_identifiers(raw_df)
        _, iterations = anon.k_anonymize(df_qi)
        assert len(iterations) >= 1

    def test_precision_in_output(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2)
        df_qi = anon.compute_quasi_identifiers(raw_df)
        df_k, _ = anon.k_anonymize(df_qi)
        if not df_k.empty:
            assert "applied_geohash_precision" in df_k.columns

    def test_k_equals_2(self, raw_df):
        anon3 = TransportDataAnonymizer(k=3, l=2)
        anon2 = TransportDataAnonymizer(k=2, l=2)
        df_qi3 = anon3.compute_quasi_identifiers(raw_df)
        df_qi2 = anon2.compute_quasi_identifiers(raw_df)
        df_k3, _ = anon3.k_anonymize(df_qi3)
        df_k2, _ = anon2.k_anonymize(df_qi2)
        assert len(df_k2) >= len(df_k3)


class TestLDiversity:

    def _make_homogeneous_df(self) -> pd.DataFrame:
        rows = []
        for i in range(10):
            rows.append({
                "qi_geohash": "u8c5h",
                "qi_time_bucket": 0,
                "qi_speed_cat": "low",
                "sa_behavior": "normal",
                "score_total": 85.0,
                "driver_pseudonym": "DRV_TEST",
                "applied_geohash_precision": 5,
            })
        for i in range(5):
            rows.append({
                "qi_geohash": "u8c5j",
                "qi_time_bucket": 1,
                "qi_speed_cat": "low",
                "sa_behavior": "normal" if i < 2 else "drowsy",
                "score_total": 75.0,
                "driver_pseudonym": "DRV_TEST2",
                "applied_geohash_precision": 5,
            })
        return pd.DataFrame(rows)

    def test_l_diversity_removes_homogeneous_class(self):
        anon = TransportDataAnonymizer(k=2, l=2)
        df = self._make_homogeneous_df()
        df_l, stats = anon.l_diversify(df)
        assert len(df_l) < len(df)

    def test_all_classes_satisfy_l(self, raw_df):
        l = 2
        anon = TransportDataAnonymizer(k=3, l=l)
        df_qi = anon.compute_quasi_identifiers(raw_df)
        df_k, _ = anon.k_anonymize(df_qi)
        df_l, _ = anon.l_diversify(df_k)

        if not df_l.empty:
            diversity = df_l.groupby(
                ["qi_geohash", "qi_time_bucket", "qi_speed_cat"]
            )["sa_behavior"].nunique()
            assert (diversity >= l).all()

    def test_stats_returned(self):
        anon = TransportDataAnonymizer(k=2, l=2)
        df = self._make_homogeneous_df()
        _, stats = anon.l_diversify(df)
        for key in ("total_classes", "satisfying_classes",
                    "suppressed_records", "kept_records"):
            assert key in stats

    def test_l_equals_3_more_suppression(self, raw_df):
        anon2 = TransportDataAnonymizer(k=3, l=2)
        anon3 = TransportDataAnonymizer(k=3, l=3)

        df_qi2 = anon2.compute_quasi_identifiers(raw_df)
        df_k2, _ = anon2.k_anonymize(df_qi2)
        df_l2, _ = anon2.l_diversify(df_k2)

        df_qi3 = anon3.compute_quasi_identifiers(raw_df)
        df_k3, _ = anon3.k_anonymize(df_qi3)
        df_l3, _ = anon3.l_diversify(df_k3)

        assert len(df_l3) <= len(df_l2)


class TestAnonymizePipeline:

    def test_returns_dataframe_and_report(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2)
        df_anon, report = anon.anonymize(raw_df)
        assert isinstance(df_anon, pd.DataFrame)
        assert isinstance(report, AnonymizationReport)

    def test_direct_identifiers_removed(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2)
        df_anon, _ = anon.anonymize(raw_df)
        for col in ("gps_latitude", "gps_longitude", "timestamp_sec", "trip_id"):
            assert col not in df_anon.columns

    def test_report_counts_consistent(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2)
        df_anon, report = anon.anonymize(raw_df)
        assert report.original_records == len(raw_df)
        assert report.anonymized_records == len(df_anon)
        assert report.suppressed_records == report.original_records - report.anonymized_records

    def test_suppression_rate_between_0_and_1(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2)
        _, report = anon.anonymize(raw_df)
        assert 0.0 <= report.suppression_rate <= 1.0

    def test_qi_columns_present_in_output(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2)
        df_anon, _ = anon.anonymize(raw_df)
        for col in ("qi_geohash", "qi_time_bucket", "qi_speed_cat", "sa_behavior"):
            assert col in df_anon.columns

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            TransportDataAnonymizer(k=1, l=2)

    def test_invalid_l_raises(self):
        with pytest.raises(ValueError):
            TransportDataAnonymizer(k=3, l=1)

    def test_report_summary_is_string(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2)
        _, report = anon.anonymize(raw_df)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "k=" in summary
        assert "l=" in summary

    def test_report_to_dict(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2)
        _, report = anon.anonymize(raw_df)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["k_parameter"] == 3
        assert d["l_parameter"] == 2

    def test_anonymized_records_le_original(self, raw_df):
        anon = TransportDataAnonymizer(k=3, l=2)
        df_anon, report = anon.anonymize(raw_df)
        assert len(df_anon) <= len(raw_df)

    def test_high_k_more_suppression(self, raw_df):
        anon3  = TransportDataAnonymizer(k=3,  l=2)
        anon10 = TransportDataAnonymizer(k=10, l=2)
        df3,  _ = anon3.anonymize(raw_df)
        df10, _ = anon10.anonymize(raw_df)
        assert len(df10) <= len(df3)
