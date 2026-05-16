"""
Інтеграційні тести src.pipeline
Покриття: повний пайплайн на синтетичному датасеті,
          коректність вихідних файлів.
"""

import json
import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import AnonymizationPipeline


@pytest.fixture()
def pipeline(synthetic_dataset, tmp_path):
    return AnonymizationPipeline(
        dataset_path=str(synthetic_dataset),
        output_path=str(tmp_path / "output"),
        k=3,
        l=2,
        geohash_precision=5,
        time_bucket_sec=300,
    )


class TestPipelineRun:

    def test_run_completes(self, pipeline):
        report = pipeline.run()
        assert report is not None

    def test_report_has_correct_params(self, pipeline):
        report = pipeline.run()
        assert report.k_parameter == 3
        assert report.l_parameter == 2

    def test_original_records_count(self, pipeline):
        report = pipeline.run()
        assert report.original_records > 0

    def test_suppression_rate_acceptable(self, pipeline):
        report = pipeline.run()
        assert report.suppression_rate < 0.5

    def test_anonymized_le_original(self, pipeline):
        report = pipeline.run()
        assert report.anonymized_records <= report.original_records


class TestPipelineOutputFiles:

    @pytest.fixture(autouse=True)
    def run_pipeline(self, pipeline, tmp_path):
        self.output_dir = Path(str(tmp_path / "output"))
        pipeline.run()

    def test_csv_created(self):
        assert (self.output_dir / "anonymized_data.csv").exists()

    def test_owl_created(self):
        assert (self.output_dir / "transport_anonymized.owl").exists()

    def test_ttl_created(self):
        assert (self.output_dir / "transport_anonymized.ttl").exists()

    def test_report_json_created(self):
        assert (self.output_dir / "anonymization_report.json").exists()

    def test_trips_summary_csv_created(self):
        assert (self.output_dir / "trips_summary.csv").exists()

    def test_csv_not_empty(self):
        df = pd.read_csv(self.output_dir / "anonymized_data.csv")
        assert len(df) > 0

    def test_csv_no_gps_columns(self):
        df = pd.read_csv(self.output_dir / "anonymized_data.csv")
        assert "gps_latitude"  not in df.columns
        assert "gps_longitude" not in df.columns

    def test_csv_has_qi_columns(self):
        df = pd.read_csv(self.output_dir / "anonymized_data.csv")
        for col in ("qi_geohash", "qi_time_bucket", "qi_speed_cat", "sa_behavior"):
            assert col in df.columns

    def test_report_json_valid(self):
        with open(self.output_dir / "anonymization_report.json", encoding="utf-8") as f:
            data = json.load(f)
        assert "k_parameter" in data
        assert "l_parameter" in data
        assert "original_records" in data
        assert "suppression_rate" in data
        assert "anonymized_records" in data

    def test_owl_parseable(self):
        from rdflib import Graph
        g = Graph()
        g.parse(str(self.output_dir / "transport_anonymized.owl"), format="xml")
        assert len(g) > 0

    def test_trips_summary_has_expected_columns(self):
        df = pd.read_csv(self.output_dir / "trips_summary.csv")
        assert "driver_pseudonym" in df.columns
        assert "behavior" in df.columns
        assert "road_type" in df.columns


class TestPipelineEdgeCases:

    def test_invalid_dataset_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AnonymizationPipeline(
                dataset_path=str(tmp_path / "nonexistent"),
                output_path=str(tmp_path / "out"),
            )

    def test_high_k_still_runs(self, synthetic_dataset, tmp_path):
        p = AnonymizationPipeline(
            dataset_path=str(synthetic_dataset),
            output_path=str(tmp_path / "out_k50"),
            k=50, l=2,
        )
        report = p.run()
        assert report.suppression_rate <= 1.0

    def test_different_k_l_params(self, synthetic_dataset, tmp_path):
        p3 = AnonymizationPipeline(
            dataset_path=str(synthetic_dataset),
            output_path=str(tmp_path / "out_k3"),
            k=3, l=2,
        )
        p2 = AnonymizationPipeline(
            dataset_path=str(synthetic_dataset),
            output_path=str(tmp_path / "out_k2"),
            k=2, l=2,
        )
        r3 = p3.run()
        r2 = p2.run()
        assert r2.anonymized_records >= r3.anonymized_records
