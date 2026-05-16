"""
OWL-онтологія для анонімізованих транспортних даних.

Технологічний стек:
  - rdflib   — побудова RDF/OWL графу, SPARQL-запити
  - owlready2 — опціонально для логічного висновку

Онтологія описує:
  Класи:
    TransportRecord, AnonymizedRecord, GPSCell, TimeBucket,
    BehaviorCategory (NormalDriving, DrowsyDriving, AggressiveDriving),
    EquivalenceClass, Driver, Trip

  Властивості:
    hasGPSCell, hasTimeBucket, hasBehavior, belongsToEquivalenceClass,
    isPartOfTrip, performedBy
    hasGeohash, hasBucketIndex, hasScoreTotal,
    hasSpeedCategory, hasDriverPseudonym,
    hasRoadType, hasClassSize,
    hasKParameter, hasLParameter
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from rdflib import (
    XSD, Graph, Literal, Namespace,
    OWL, RDF, RDFS, URIRef,
)

logger = logging.getLogger(__name__)

T = Namespace("http://uah.edu.ua/ontologies/transport#")
A = Namespace("http://uah.edu.ua/ontologies/anonymization#")

class TransportAnonymizationOntology:
    """ OWL-онтологія анонімізованих транспортних даних. """

    def __init__(self) -> None:
        self.g = Graph()
        self.g.bind("t", T)
        self.g.bind("anon", A)
        self.g.bind("owl", OWL)
        self.g.bind("rdfs", RDFS)
        self.g.bind("xsd", XSD)
        self._define_schema()


    def _cls(self, uri: URIRef, label_uk: str, comment_uk: str = "",
             parent: Optional[URIRef] = None) -> None:
        self.g.add((uri, RDF.type, OWL.Class))
        self.g.add((uri, RDFS.label, Literal(label_uk, lang="uk")))
        if comment_uk:
            self.g.add((uri, RDFS.comment, Literal(comment_uk, lang="uk")))
        if parent:
            self.g.add((uri, RDFS.subClassOf, parent))

    def _obj_prop(self, uri: URIRef, label_uk: str,
                  domain: URIRef, range_uri: URIRef) -> None:
        self.g.add((uri, RDF.type, OWL.ObjectProperty))
        self.g.add((uri, RDFS.label, Literal(label_uk, lang="uk")))
        self.g.add((uri, RDFS.domain, domain))
        self.g.add((uri, RDFS.range, range_uri))

    def _data_prop(self, uri: URIRef, label_uk: str,
                   domain: URIRef, xsd_range: URIRef) -> None:
        self.g.add((uri, RDF.type, OWL.DatatypeProperty))
        self.g.add((uri, RDFS.label, Literal(label_uk, lang="uk")))
        self.g.add((uri, RDFS.domain, domain))
        self.g.add((uri, RDFS.range, xsd_range))

    def _define_schema(self) -> None:
        """Оголошує всі класи та властивості онтології."""

        onto_uri = URIRef("http://uah.edu.ua/ontologies/transport-anonymization.owl")
        self.g.add((onto_uri, RDF.type, OWL.Ontology))
        self.g.add((onto_uri, RDFS.label,
                    Literal("Онтологія анонімізації транспортних даних", lang="uk")))
        self.g.add((onto_uri, RDFS.comment,
                    Literal("Магістерська робота. k-анонімність + l-різноманітність",
                            lang="uk")))

        self._cls(T.TransportRecord, "Транспортний запис",
                  "Телематичний запис однієї точки маршруту")

        self._cls(T.AnonymizedRecord, "Анонімізований запис",
                  "Запис після k-анонімності та l-різноманітності",
                  parent=T.TransportRecord)

        self._cls(T.GPSCell, "Просторова комірка GPS",
                  "Геохеш-комірка, що замінює точні координати")

        self._cls(T.TimeBucket, "Часовий інтервал",
                  "Дискретний часовий bucket замість точної мітки часу")

        self._cls(T.BehaviorCategory, "Категорія поведінки водія",
                  "normal / drowsy / aggressive")

        self._cls(T.NormalDriving, "Нормальне водіння",
                  parent=T.BehaviorCategory)
        self._cls(T.DrowsyDriving, "Сонливе водіння",
                  parent=T.BehaviorCategory)
        self._cls(T.AggressiveDriving, "Агресивне водіння",
                  parent=T.BehaviorCategory)

        self._cls(T.EquivalenceClass, "Клас еквівалентності",
                  "Група записів з однаковими QI (k-анонімність)")

        self._cls(T.Driver, "Водій (псевдонімізований)",
                  "Анонімізований водій без прямих ідентифікаторів")

        self._cls(T.Trip, "Поїздка",
                  "Один маршрут з метаданими (дорога, поведінка)")

        self._obj_prop(T.hasGPSCell, "має просторову комірку",
                       T.TransportRecord, T.GPSCell)
        self._obj_prop(T.hasTimeBucket, "належить до часового інтервалу",
                       T.TransportRecord, T.TimeBucket)
        self._obj_prop(T.hasBehavior, "має поведінку",
                       T.TransportRecord, T.BehaviorCategory)
        self._obj_prop(T.belongsToEquivalenceClass, "належить до класу еквівалентності",
                       T.AnonymizedRecord, T.EquivalenceClass)
        self._obj_prop(T.isPartOfTrip, "є частиною поїздки",
                       T.TransportRecord, T.Trip)
        self._obj_prop(T.performedBy, "виконана водієм",
                       T.Trip, T.Driver)

        self._data_prop(T.hasGeohash, "геохеш комірки",
                        T.GPSCell, XSD.string)
        self._data_prop(T.hasPrecision, "точність геохешу",
                        T.GPSCell, XSD.integer)
        self._data_prop(T.hasBucketIndex, "індекс bucket",
                        T.TimeBucket, XSD.integer)
        self._data_prop(T.hasBucketSizeSeconds, "розмір bucket (секунди)",
                        T.TimeBucket, XSD.integer)
        self._data_prop(T.hasScoreTotal, "загальна оцінка водіння",
                        T.TransportRecord, XSD.float)
        self._data_prop(T.hasSpeedCategory, "категорія швидкості",
                        T.TransportRecord, XSD.string)
        self._data_prop(T.hasDriverPseudonym, "псевдонім водія",
                        T.Driver, XSD.string)
        self._data_prop(T.hasRoadType, "тип дороги",
                        T.Trip, XSD.string)
        self._data_prop(T.hasClassSize, "розмір класу еквівалентності",
                        T.EquivalenceClass, XSD.integer)
        self._data_prop(T.hasKParameter, "параметр k",
                        T.EquivalenceClass, XSD.integer)
        self._data_prop(T.hasLParameter, "параметр l",
                        T.EquivalenceClass, XSD.integer)

    def _behavior_uri(self, behavior: str) -> URIRef:
        return {
            "normal": T.NormalDriving,
            "drowsy": T.DrowsyDriving,
            "aggressive": T.AggressiveDriving,
        }.get(behavior.lower(), T.BehaviorCategory)

    def populate_from_dataframe(
        self,
        df: pd.DataFrame,
        k: int = 3,
        l: int = 2,
        bucket_size_sec: int = 300,
    ) -> "TransportAnonymizationOntology":
        """
        Заповнює онтологію індивідами з анонімізованого DataFrame.

        Ідентичні QI-комбінації отримують один і той же GPSCell /
        TimeBucket / EquivalenceClass.
        """
        gps_cells: Dict[str, URIRef] = {}
        time_buckets: Dict[int, URIRef] = {}
        eq_classes: Dict[str, URIRef] = {}
        drivers: Dict[str, URIRef] = {}
        eq_class_records: Dict[str, int] = {}

        for idx, row in df.iterrows():
            geohash = str(row.get("qi_geohash", "unknown"))
            time_bucket = int(row.get("qi_time_bucket", -1))
            speed_cat = str(row.get("qi_speed_cat", "unknown"))
            behavior = str(row.get("sa_behavior", "unknown"))
            pseudonym = str(row.get("driver_pseudonym", "UNKNOWN"))
            precision = int(row.get("applied_geohash_precision", 5))

            if geohash not in gps_cells:
                cell_uri = T[f"cell_{geohash}"]
                self.g.add((cell_uri, RDF.type, T.GPSCell))
                self.g.add((cell_uri, T.hasGeohash,
                            Literal(geohash, datatype=XSD.string)))
                self.g.add((cell_uri, T.hasPrecision,
                            Literal(precision, datatype=XSD.integer)))
                gps_cells[geohash] = cell_uri

            if time_bucket not in time_buckets:
                bucket_uri = T[f"bucket_{time_bucket}"]
                self.g.add((bucket_uri, RDF.type, T.TimeBucket))
                self.g.add((bucket_uri, T.hasBucketIndex,
                            Literal(time_bucket, datatype=XSD.integer)))
                self.g.add((bucket_uri, T.hasBucketSizeSeconds,
                            Literal(bucket_size_sec, datatype=XSD.integer)))
                time_buckets[time_bucket] = bucket_uri

            eq_key = f"{geohash}|{time_bucket}|{speed_cat}"
            eq_id = f"eqclass_{abs(hash(eq_key)) % 10**10}"
            if eq_key not in eq_classes:
                eq_uri = T[eq_id]
                self.g.add((eq_uri, RDF.type, T.EquivalenceClass))
                self.g.add((eq_uri, T.hasKParameter,
                            Literal(k, datatype=XSD.integer)))
                self.g.add((eq_uri, T.hasLParameter,
                            Literal(l, datatype=XSD.integer)))
                eq_classes[eq_key] = eq_uri
                eq_class_records[eq_key] = 0

            drv_short = pseudonym[-8:]
            if drv_short not in drivers:
                drv_uri = T[f"driver_{drv_short}"]
                self.g.add((drv_uri, RDF.type, T.Driver))
                self.g.add((drv_uri, T.hasDriverPseudonym,
                            Literal(pseudonym, datatype=XSD.string)))
                drivers[drv_short] = drv_uri

            rec_uri = T[f"record_{idx}"]
            self.g.add((rec_uri, RDF.type, T.AnonymizedRecord))
            self.g.add((rec_uri, T.hasGPSCell, gps_cells[geohash]))
            self.g.add((rec_uri, T.hasTimeBucket, time_buckets[time_bucket]))
            self.g.add((rec_uri, T.hasBehavior, self._behavior_uri(behavior)))
            self.g.add((rec_uri, T.belongsToEquivalenceClass, eq_classes[eq_key]))
            self.g.add((rec_uri, T.hasSpeedCategory,
                        Literal(speed_cat, datatype=XSD.string)))

            score = row.get("score_total")
            if pd.notna(score):
                self.g.add((rec_uri, T.hasScoreTotal,
                            Literal(float(score), datatype=XSD.float)))

            eq_class_records[eq_key] += 1

        for eq_key, count in eq_class_records.items():
            self.g.set((eq_classes[eq_key], T.hasClassSize,
                        Literal(count, datatype=XSD.integer)))

        logger.info(
            "Онтологія заповнена: %d триплетів, %d класів еквів.",
            len(self.g), len(eq_classes),
        )
        return self

    def save(self, filepath: str, fmt: str = "xml") -> "TransportAnonymizationOntology":
        """ Зберігає онтологію у файл. """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.g.serialize(destination=filepath, format=fmt)
        logger.info("Онтологія збережена: %s (%s)", filepath, fmt)
        return self

    def sparql(self, query: str) -> pd.DataFrame:
        """Виконує SPARQL-запит, повертає DataFrame."""
        results = self.g.query(query)
        rows = [
            {k: (str(v) if v is not None else None) for k, v in row.asdict().items()}
            for row in results
        ]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def behavior_distribution(self) -> pd.DataFrame:
        """Розподіл анонімізованих записів за типом поведінки."""
        return self.sparql("""
            PREFIX t: <http://uah.edu.ua/ontologies/transport#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?behavior (COUNT(?rec) AS ?count)
            WHERE {
                ?rec a t:AnonymizedRecord ;
                     t:hasBehavior ?bClass .
                ?bClass rdfs:label ?behavior .
                FILTER(LANG(?behavior) = "uk")
            }
            GROUP BY ?behavior
            ORDER BY DESC(?count)
        """)

    def equivalence_class_stats(self, limit: int = 10) -> pd.DataFrame:
        """Топ класів еквівалентності за розміром."""
        return self.sparql(f"""
            PREFIX t: <http://uah.edu.ua/ontologies/transport#>

            SELECT ?class ?size ?k ?l
            WHERE {{
                ?class a t:EquivalenceClass ;
                       t:hasClassSize ?size ;
                       t:hasKParameter ?k ;
                       t:hasLParameter ?l .
            }}
            ORDER BY DESC(?size)
            LIMIT {limit}
        """)

    def geohash_cell_counts(self) -> pd.DataFrame:
        """Кількість записів по кожній геохеш-комірці."""
        return self.sparql("""
            PREFIX t: <http://uah.edu.ua/ontologies/transport#>

            SELECT ?geohash (COUNT(?rec) AS ?count)
            WHERE {
                ?rec a t:AnonymizedRecord ;
                     t:hasGPSCell ?cell .
                ?cell t:hasGeohash ?geohash .
            }
            GROUP BY ?geohash
            ORDER BY DESC(?count)
        """)
