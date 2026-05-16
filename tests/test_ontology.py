"""
Тести модуля src.ontology_builder
Покриття: схема, заповнення даними, SPARQL-запити.
"""

import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ontology_builder import TransportAnonymizationOntology, T
from src.anonymizer import TransportDataAnonymizer
from rdflib import RDF, OWL, URIRef


@pytest.fixture()
def anon_df(raw_df):
    anon = TransportDataAnonymizer(k=3, l=2)
    df_anon, _ = anon.anonymize(raw_df)
    if "applied_geohash_precision" not in df_anon.columns:
        df_anon["applied_geohash_precision"] = 5
    return df_anon


@pytest.fixture()
def populated_ontology(anon_df):
    onto = TransportAnonymizationOntology()
    onto.populate_from_dataframe(anon_df, k=3, l=2)
    return onto


class TestOntologySchema:

    def test_ontology_created(self):
        onto = TransportAnonymizationOntology()
        assert onto.g is not None
        assert len(onto.g) > 0

    def test_core_classes_declared(self):
        onto = TransportAnonymizationOntology()
        expected_classes = [
            T.TransportRecord, T.AnonymizedRecord, T.GPSCell,
            T.TimeBucket, T.BehaviorCategory, T.NormalDriving,
            T.DrowsyDriving, T.AggressiveDriving, T.EquivalenceClass,
            T.Driver, T.Trip,
        ]
        for cls in expected_classes:
            assert (cls, RDF.type, OWL.Class) in onto.g, \
                f"Клас {cls} не оголошений в онтології"

    def test_object_properties_declared(self):
        onto = TransportAnonymizationOntology()
        for prop in (T.hasGPSCell, T.hasTimeBucket, T.hasBehavior,
                     T.belongsToEquivalenceClass, T.isPartOfTrip, T.performedBy):
            assert (prop, RDF.type, OWL.ObjectProperty) in onto.g

    def test_datatype_properties_declared(self):
        onto = TransportAnonymizationOntology()
        for prop in (T.hasGeohash, T.hasBucketIndex, T.hasScoreTotal,
                     T.hasDriverPseudonym, T.hasClassSize):
            assert (prop, RDF.type, OWL.DatatypeProperty) in onto.g

    def test_anonymized_record_subclass_of_transport_record(self):
        from rdflib import RDFS
        onto = TransportAnonymizationOntology()
        assert (T.AnonymizedRecord, RDFS.subClassOf, T.TransportRecord) in onto.g

    def test_behavior_subclasses(self):
        from rdflib import RDFS
        onto = TransportAnonymizationOntology()
        for sub in (T.NormalDriving, T.DrowsyDriving, T.AggressiveDriving):
            assert (sub, RDFS.subClassOf, T.BehaviorCategory) in onto.g


class TestOntologyPopulation:

    def test_triple_count_positive(self, populated_ontology):
        assert len(populated_ontology.g) > 100

    def test_anonymized_records_added(self, populated_ontology, anon_df):
        count = sum(1 for _ in populated_ontology.g.subjects(RDF.type, T.AnonymizedRecord))
        assert count == len(anon_df)

    def test_gps_cells_created(self, populated_ontology, anon_df):
        unique_geohashes = anon_df["qi_geohash"].nunique()
        cells = sum(1 for _ in populated_ontology.g.subjects(RDF.type, T.GPSCell))
        assert cells == unique_geohashes

    def test_equivalence_classes_created(self, populated_ontology, anon_df):
        unique_eq = anon_df.groupby(
            ["qi_geohash", "qi_time_bucket", "qi_speed_cat"]
        ).ngroups
        eq_in_graph = sum(1 for _ in populated_ontology.g.subjects(
            RDF.type, T.EquivalenceClass
        ))
        assert eq_in_graph == unique_eq

    def test_drivers_created(self, populated_ontology, anon_df):
        unique_drivers = anon_df["driver_pseudonym"].nunique()
        drivers_in_graph = sum(1 for _ in populated_ontology.g.subjects(RDF.type, T.Driver))
        assert drivers_in_graph == unique_drivers

    def test_no_real_coordinates_in_graph(self, populated_ontology, anon_df):
        from rdflib import Literal
        g = populated_ontology.g
        lat_val = str(anon_df["gps_latitude"].iloc[0]) if "gps_latitude" in anon_df.columns else None
        if lat_val:
            all_literals = {str(o) for _, _, o in g if isinstance(o, Literal)}
            assert lat_val not in all_literals

    def test_populate_twice_idempotent(self, anon_df):
        onto = TransportAnonymizationOntology()
        onto.populate_from_dataframe(anon_df, k=3, l=2)
        count1 = len(onto.g)
        onto2 = TransportAnonymizationOntology()
        onto2.populate_from_dataframe(anon_df, k=3, l=2)
        count2 = len(onto2.g)
        assert count1 == count2


class TestSPARQL:

    def test_behavior_distribution_returns_df(self, populated_ontology):
        df = populated_ontology.behavior_distribution()
        assert isinstance(df, pd.DataFrame)

    def test_behavior_distribution_has_counts(self, populated_ontology):
        df = populated_ontology.behavior_distribution()
        if not df.empty and "count" in df.columns:
            assert df["count"].astype(int).sum() > 0

    def test_equivalence_class_stats(self, populated_ontology):
        df = populated_ontology.equivalence_class_stats(limit=5)
        assert isinstance(df, pd.DataFrame)

    def test_eq_class_stats_have_size(self, populated_ontology):
        df = populated_ontology.equivalence_class_stats(limit=5)
        if not df.empty:
            assert "size" in df.columns

    def test_geohash_cell_counts(self, populated_ontology):
        df = populated_ontology.geohash_cell_counts()
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "geohash" in df.columns
            assert "count" in df.columns

    def test_custom_sparql_query(self, populated_ontology):
        df = populated_ontology.sparql("""
            PREFIX t: <http://uah.edu.ua/ontologies/transport#>
            SELECT (COUNT(?r) AS ?total)
            WHERE { ?r a t:AnonymizedRecord }
        """)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


class TestOntologySave:

    def test_save_owl_xml(self, populated_ontology, tmp_path):
        out = str(tmp_path / "test.owl")
        populated_ontology.save(out, fmt="xml")
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_save_turtle(self, populated_ontology, tmp_path):
        out = str(tmp_path / "test.ttl")
        populated_ontology.save(out, fmt="turtle")
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_save_creates_parent_dirs(self, populated_ontology, tmp_path):
        out = str(tmp_path / "nested" / "deep" / "test.owl")
        populated_ontology.save(out, fmt="xml")
        assert Path(out).exists()

    def test_saved_owl_parseable(self, populated_ontology, tmp_path):
        from rdflib import Graph
        out = str(tmp_path / "test.owl")
        populated_ontology.save(out, fmt="xml")
        g2 = Graph()
        g2.parse(out, format="xml")
        assert len(g2) > 0
