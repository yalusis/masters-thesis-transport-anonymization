"""
Головний пайплайн анонімізації транспортних даних UAH-DriveSet.

Етапи:
  1. Завантаження датасету  (UAHDataLoader)
  2. Анонімізація           (TransportDataAnonymizer → k-ан. + l-різн.)
  3. Збереження CSV         (anonymized_data.csv)
  4. Побудова онтології     (TransportAnonymizationOntology)
  5. Збереження OWL/Turtle  (transport_anonymized.owl / .ttl)
  6. Звіт JSON              (anonymization_report.json)
  7. SPARQL-демонстрація
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

import pandas as pd

from src.data_loader import UAHDataLoader
from src.anonymizer import TransportDataAnonymizer, AnonymizationReport
from src.ontology_builder import TransportAnonymizationOntology
from config import K_ANONYMITY, L_DIVERSITY, GEOHASH_PRECISION, TIME_BUCKET_SECONDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline")


class AnonymizationPipeline:
    """ Оркестратор повного пайплайну. """

    def __init__(
        self,
        dataset_path: str,
        output_path: str = "./output",
        k: int = K_ANONYMITY,
        l: int = L_DIVERSITY,
        geohash_precision: int = GEOHASH_PRECISION,
        time_bucket_sec: int = TIME_BUCKET_SECONDS,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.k = k
        self.l = l
        self.geohash_precision = geohash_precision
        self.time_bucket_sec = time_bucket_sec

        self.loader = UAHDataLoader(str(self.dataset_path))
        self.anonymizer = TransportDataAnonymizer(
            k=k, l=l,
            geohash_precision=geohash_precision,
            time_bucket_sec=time_bucket_sec,
        )
        self.ontology = TransportAnonymizationOntology()

    def run(self) -> AnonymizationReport:
        """Запускає повний пайплайн і повертає звіт."""

        logger.info("=" * 60)
        logger.info("  ПАЙПЛАЙН АНОНІМІЗАЦІЇ ТРАНСПОРТНИХ ДАНИХ")
        logger.info("=" * 60)
        logger.info("Датасет:  %s", self.dataset_path)
        logger.info("Вихід:    %s", self.output_path)
        logger.info("Параметри: k=%d, l=%d, geohash_prec=%d, bucket=%ds",
                    self.k, self.l, self.geohash_precision, self.time_bucket_sec)

        logger.info("\n[1/5] Завантаження датасету...")
        df_raw, finals = self.loader.load_full_dataset()
        logger.info("  Завантажено: %d записів | %d поїздок", len(df_raw), len(finals))

        trips_summary = pd.DataFrame([
            {
                "driver_pseudonym": f["driver_pseudonym"],
                "behavior": f["meta"]["behavior"],
                "road_type": f["meta"]["road_type"],
                "distance_km": f["meta"]["distance_km"],
                **{k: v for k, v in f["final"].items()
                   if k in ("avg_speed_kmh", "max_speed_kmh",
                            "driving_time_min", "score_total_final")},
            }
            for f in finals
        ])
        trips_summary.to_csv(self.output_path / "trips_summary.csv", index=False)

        logger.info("\n[2/5] Анонімізація (k-anonymity + l-diversity)...")
        df_anon, report = self.anonymizer.anonymize(df_raw)

        logger.info("\n[3/5] Збереження анонімізованих даних...")
        csv_path = self.output_path / "anonymized_data.csv"
        df_anon.to_csv(csv_path, index=False)
        logger.info("  CSV: %s (%d рядків, %d стовпців)",
                    csv_path, len(df_anon), len(df_anon.columns))

        logger.info("\n[4/5] Побудова OWL-онтології...")
        self.ontology.populate_from_dataframe(
            df_anon, k=self.k, l=self.l,
            bucket_size_sec=self.time_bucket_sec,
        )

        owl_path = self.output_path / "transport_anonymized.owl"
        ttl_path = self.output_path / "transport_anonymized.ttl"
        self.ontology.save(str(owl_path), fmt="xml")
        self.ontology.save(str(ttl_path), fmt="turtle")
        logger.info("  OWL: %s", owl_path)
        logger.info("  TTL: %s", ttl_path)
        logger.info("  Триплетів у графі: %d", len(self.ontology.g))

        logger.info("\n[5/5] Формування звіту...")
        report_dict = {
            **report.to_dict(),
            "generated_at": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_path),
            "output_path": str(self.output_path),
            "ontology_triples": len(self.ontology.g),
            "trips_loaded": len(finals),
            "columns_anonymized": list(df_anon.columns),
        }
        report_path = self.output_path / "anonymization_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        logger.info("  Звіт: %s", report_path)

        logger.info("\n── SPARQL-запити до онтології ──")

        behavior_df = self.ontology.behavior_distribution()
        if not behavior_df.empty:
            logger.info("Розподіл поведінки:\n%s", behavior_df.to_string(index=False))

        eq_df = self.ontology.equivalence_class_stats(limit=5)
        if not eq_df.empty:
            logger.info("Топ-5 класів еквівалентності:\n%s", eq_df.to_string(index=False))

        geohash_df = self.ontology.geohash_cell_counts()
        if not geohash_df.empty:
            logger.info("Топ-5 GPS-комірок:\n%s", geohash_df.head(5).to_string(index=False))

        logger.info("\n" + "=" * 60)
        logger.info("  ПАЙПЛАЙН ЗАВЕРШЕНО УСПІШНО")
        logger.info("  Збережено в: %s", self.output_path)
        logger.info("=" * 60)

        return report
