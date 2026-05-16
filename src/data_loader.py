"""
Data Loader
Завантаження та парсинг файлів датасету:
  - SEMANTIC_ONLINE.txt
  - SEMANTIC_FINAL.txt
"""

import re
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import ONLINE_COLUMNS, FINAL_FIELDS

logger = logging.getLogger(__name__)

_FOLDER_RE = re.compile(
    r"(\d{14})-(\d+\.?\d*)km?-(\w+)-(NORMAL\d?|DROWSY\d?|AGGRESSIVE\d?)-(MOTORWAY|SECONDARY)",
    re.IGNORECASE,
)


class UAHDataLoader:
    """ Завантажувач датасету. """

    def __init__(self, dataset_root: str) -> None:
        self.root = Path(dataset_root)
        if not self.root.exists():
            raise FileNotFoundError(f"Директорія датасету не знайдена: {self.root}")

    @staticmethod
    def _parse_folder_name(name: str) -> Optional[Dict]:
        """
        Парсить метадані з назви папки поїздки.

        Returns None якщо назва не відповідає шаблону.
        """
        m = _FOLDER_RE.match(name)
        if not m:
            return None
        dt, dist, driver_id, behavior, road = m.groups()
        return {
            "datetime_str": dt,
            "distance_km": float(dist),
            "driver_id": driver_id,
            "behavior": behavior.rstrip("0123456789").lower(),
            "road_type": road.lower(),
            "trip_id": f"{driver_id}_{dt}",
        }

    @staticmethod
    def _pseudonymize(driver_id: str) -> str:
        """
        Псевдонімізує ідентифікатор водія через SHA-256.
        Прямий driver_id не зберігається ніде у вихідних даних.
        """
        digest = hashlib.sha256(driver_id.encode("utf-8")).hexdigest()[:12].upper()
        return f"DRV_{digest}"

    def load_semantic_online(self, filepath: Path) -> pd.DataFrame:
        """
        Завантажує SEMANTIC_ONLINE.txt.
        Рядки з некоректними координатами відфільтровуються.
        """
        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            names=ONLINE_COLUMNS,
            comment="#",
        )
        df = df.apply(pd.to_numeric, errors="coerce")

        valid_gps = (
            df["gps_latitude"].notna()
            & df["gps_longitude"].notna()
            & df["gps_latitude"].between(-90, 90)
            & df["gps_longitude"].between(-180, 180)
        )
        dropped = (~valid_gps).sum()
        if dropped:
            logger.debug("  Відфільтровано %d рядків без валідних GPS", dropped)
        return df[valid_gps].reset_index(drop=True)

    def load_semantic_final(self, filepath: Path) -> Dict:
        """Завантажує SEMANTIC_FINAL.txt."""
        lines = [
            ln.strip()
            for ln in filepath.read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        result: Dict = {}
        for idx, field_name in FINAL_FIELDS.items():
            if idx < len(lines):
                try:
                    result[field_name] = float(lines[idx])
                except ValueError:
                    result[field_name] = None
        return result

    def load_trip(self, trip_folder: Path) -> Optional[Dict]:
        """
        Завантажує одну поїздку (online + final).

        Returns None якщо SEMANTIC_ONLINE.txt відсутній або папка
        не відповідає формату датасету.
        """
        meta = self._parse_folder_name(trip_folder.name)
        if meta is None:
            return None

        online_file = trip_folder / "SEMANTIC_ONLINE.txt"
        final_file = trip_folder / "SEMANTIC_FINAL.txt"

        if not online_file.exists():
            logger.warning("SEMANTIC_ONLINE.txt відсутній: %s", trip_folder)
            return None

        online_df = self.load_semantic_online(online_file)
        if online_df.empty:
            logger.warning("Порожній SEMANTIC_ONLINE: %s", trip_folder)
            return None

        pseudonym = self._pseudonymize(meta["driver_id"])

        online_df["trip_id"] = meta["trip_id"]
        online_df["driver_pseudonym"] = pseudonym
        online_df["behavior_label"] = meta["behavior"]
        online_df["road_type"] = meta["road_type"]

        final_data: Dict = {}
        if final_file.exists():
            final_data = self.load_semantic_final(final_file)

        return {
            "meta": meta,
            "online_df": online_df,
            "final_data": final_data,
            "driver_pseudonym": pseudonym,
        }

    def load_driver(self, driver_folder: Path) -> List[Dict]:
        """Завантажує всі поїздки одного водія."""
        trips = []
        for sub in sorted(driver_folder.iterdir()):
            if sub.is_dir():
                trip = self.load_trip(sub)
                if trip:
                    trips.append(trip)
                    logger.info("  Завантажено: %s (%d рядків)",
                                sub.name, len(trip["online_df"]))
        return trips

    def load_full_dataset(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """ Завантажує весь датасет UAH-DriveSet. """
        all_dfs: List[pd.DataFrame] = []
        all_finals: List[Dict] = []

        for driver_folder in sorted(self.root.iterdir()):
            if not driver_folder.is_dir():
                continue
            logger.info("Завантаження водія: %s", driver_folder.name)

            for trip_folder in sorted(driver_folder.iterdir()):
                if not trip_folder.is_dir():
                    continue
                trip = self.load_trip(trip_folder)
                if trip:
                    all_dfs.append(trip["online_df"])
                    all_finals.append({
                        "meta": trip["meta"],
                        "final": trip["final_data"],
                        "driver_pseudonym": trip["driver_pseudonym"],
                    })
                else:
                    logger.debug("Пропущено (не відповідає формату): %s", trip_folder.name)

        if not all_dfs:
            raise ValueError(
                "Датасет порожній або структура папок не відповідає UAH-DriveSet. "
                "Перевірте шлях та наявність SEMANTIC_ONLINE.txt у папках поїздок."
            )

        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(
            "Датасет завантажено: %d записів, %d поїздок",
            len(combined), len(all_finals),
        )
        return combined, all_finals

