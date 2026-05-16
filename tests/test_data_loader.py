"""
Тести модуля src.data_loader
Покриття: парсинг назв папок, завантаження SEMANTIC_ONLINE/FINAL,
          псевдонімізація, обхід датасету.
"""

import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import UAHDataLoader


class TestParseFolderName:

    def test_standard_format(self):
        """Стандартний формат датасету парситься коректно."""
        meta = UAHDataLoader._parse_folder_name(
            "20151110175712-16km-D1-NORMAL-SECONDARY"
        )
        assert meta is not None
        assert meta["driver_id"] == "D1"
        assert meta["behavior"] == "normal"
        assert meta["road_type"] == "secondary"
        assert meta["distance_km"] == 16.0

    def test_behavior_with_digit(self):
        """NORMAL1 / NORMAL2 парситься і цифра відрізається."""
        meta1 = UAHDataLoader._parse_folder_name(
            "20151110175712-16km-D1-NORMAL1-SECONDARY"
        )
        meta2 = UAHDataLoader._parse_folder_name(
            "20151110180824-16km-D1-NORMAL2-SECONDARY"
        )
        assert meta1 is not None
        assert meta1["behavior"] == "normal"
        assert meta2 is not None
        assert meta2["behavior"] == "normal"

    def test_aggressive_motorway(self):
        meta = UAHDataLoader._parse_folder_name(
            "20151111125233-24km-D1-AGGRESSIVE-MOTORWAY"
        )
        assert meta is not None
        assert meta["behavior"] == "aggressive"
        assert meta["road_type"] == "motorway"

    def test_drowsy_behavior(self):
        meta = UAHDataLoader._parse_folder_name(
            "20151111132348-25km-D1-DROWSY-MOTORWAY"
        )
        assert meta is not None
        assert meta["behavior"] == "drowsy"

    def test_trip_id_format(self):
        meta = UAHDataLoader._parse_folder_name(
            "20151110175712-16km-D3-NORMAL-SECONDARY"
        )
        assert meta["trip_id"] == "D3_20151110175712"

    def test_invalid_folder_name_returns_none(self):
        """Невалідна назва повертає None."""
        assert UAHDataLoader._parse_folder_name("uah_driveset_reader") is None
        assert UAHDataLoader._parse_folder_name("random_folder") is None
        assert UAHDataLoader._parse_folder_name("") is None

    def test_decimal_distance(self):
        meta = UAHDataLoader._parse_folder_name(
            "20151110175712-12.5km-D2-NORMAL-MOTORWAY"
        )
        assert meta is not None
        assert meta["distance_km"] == 12.5


class TestPseudonymize:

    def test_returns_drv_prefix(self):
        p = UAHDataLoader._pseudonymize("D1")
        assert p.startswith("DRV_")

    def test_same_input_same_output(self):
        assert UAHDataLoader._pseudonymize("D1") == UAHDataLoader._pseudonymize("D1")

    def test_different_drivers_different_pseudonyms(self):
        assert UAHDataLoader._pseudonymize("D1") != UAHDataLoader._pseudonymize("D2")

    def test_original_id_not_in_pseudonym(self):
        p = UAHDataLoader._pseudonymize("D1")
        assert "D1" not in p


class TestLoadSemanticOnline:

    def test_loads_valid_file(self, synthetic_dataset, tmp_path):
        """Валідний файл завантажується без помилок."""
        loader = UAHDataLoader(str(synthetic_dataset))
        online_file = list(
            (synthetic_dataset / "D1").glob("*/SEMANTIC_ONLINE.txt")
        )[0]
        df = loader.load_semantic_online(online_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_column_count(self, synthetic_dataset):
        loader = UAHDataLoader(str(synthetic_dataset))
        online_file = list(
            (synthetic_dataset / "D1").glob("*/SEMANTIC_ONLINE.txt")
        )[0]
        df = loader.load_semantic_online(online_file)
        assert len(df.columns) == 27

    def test_gps_coordinates_valid(self, synthetic_dataset):
        loader = UAHDataLoader(str(synthetic_dataset))
        online_file = list(
            (synthetic_dataset / "D1").glob("*/SEMANTIC_ONLINE.txt")
        )[0]
        df = loader.load_semantic_online(online_file)
        assert df["gps_latitude"].between(-90, 90).all()
        assert df["gps_longitude"].between(-180, 180).all()

    def test_invalid_gps_filtered(self, tmp_path):
        bad_file = tmp_path / "SEMANTIC_ONLINE.txt"
        bad_file.write_text(
            "0.0 48.46 35.05 " + " ".join(["80.0"] * 24) + "\n"
            "5.0 999.0 35.05 " + " ".join(["80.0"] * 24) + "\n"
        )
        loader = UAHDataLoader(str(tmp_path.parent))
        df = loader.load_semantic_online(bad_file)
        assert len(df) == 1
        assert df["gps_latitude"].iloc[0] == pytest.approx(48.46, abs=0.001)


class TestLoadSemanticFinal:

    def test_loads_final_file(self, synthetic_dataset):
        loader = UAHDataLoader(str(synthetic_dataset))
        final_file = list(
            (synthetic_dataset / "D1").glob("*/SEMANTIC_FINAL.txt")
        )[0]
        data = loader.load_semantic_final(final_file)
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_key_fields_present(self, synthetic_dataset):
        loader = UAHDataLoader(str(synthetic_dataset))
        final_file = list(
            (synthetic_dataset / "D1").glob("*/SEMANTIC_FINAL.txt")
        )[0]
        data = loader.load_semantic_final(final_file)
        for key in ("avg_speed_kmh", "max_speed_kmh", "driving_time_min",
                    "trip_distance_km"):
            assert key in data

    def test_values_are_numeric(self, synthetic_dataset):
        loader = UAHDataLoader(str(synthetic_dataset))
        final_file = list(
            (synthetic_dataset / "D1").glob("*/SEMANTIC_FINAL.txt")
        )[0]
        data = loader.load_semantic_final(final_file)
        for v in data.values():
            assert v is None or isinstance(v, float)


class TestLoadFullDataset:

    def test_loads_all_trips(self, synthetic_dataset):
        loader = UAHDataLoader(str(synthetic_dataset))
        df, finals = loader.load_full_dataset()
        assert len(finals) == 4

    def test_returns_dataframe(self, synthetic_dataset):
        loader = UAHDataLoader(str(synthetic_dataset))
        df, _ = loader.load_full_dataset()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_metadata_columns_present(self, synthetic_dataset):
        loader = UAHDataLoader(str(synthetic_dataset))
        df, _ = loader.load_full_dataset()
        for col in ("trip_id", "driver_pseudonym", "behavior_label", "road_type"):
            assert col in df.columns

    def test_no_real_driver_ids(self, synthetic_dataset):
        loader = UAHDataLoader(str(synthetic_dataset))
        df, _ = loader.load_full_dataset()
        for real_id in ("D1", "D2"):
            assert not df["driver_pseudonym"].str.contains(real_id).any()

    def test_behaviors_loaded_correctly(self, synthetic_dataset):
        loader = UAHDataLoader(str(synthetic_dataset))
        df, _ = loader.load_full_dataset()
        behaviors = set(df["behavior_label"].unique())
        assert behaviors == {"normal", "drowsy", "aggressive"}

    def test_invalid_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            UAHDataLoader(str(tmp_path / "nonexistent"))

    def test_empty_dataset_raises(self, tmp_path):
        (tmp_path / "D1").mkdir()
        loader = UAHDataLoader(str(tmp_path))
        with pytest.raises(ValueError):
            loader.load_full_dataset()
