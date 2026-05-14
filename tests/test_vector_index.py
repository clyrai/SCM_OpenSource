"""Tests for the InMemoryVectorIndex.

Covers: add/remove/search/rebuild, idempotent add, dim mismatch,
top-k correctness, min_score threshold, empty-index search.
"""
import math
import pytest
import numpy as np

from src.retrieval.vector_index import InMemoryVectorIndex, VectorHit


def _unit(*xs):
    """Convenience: build a unit-normalized embedding from raw components."""
    arr = np.array(xs, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


def test_empty_index_returns_no_hits():
    idx = InMemoryVectorIndex()
    assert idx.search([1.0, 0.0]) == []
    assert idx.size() == 0


def test_add_and_search_returns_top_k_by_cosine():
    idx = InMemoryVectorIndex()
    idx.add("a", _unit(1.0, 0.0))
    idx.add("b", _unit(0.0, 1.0))
    idx.add("c", _unit(0.71, 0.71))
    hits = idx.search([1.0, 0.0], top_k=3)
    assert [h.concept_id for h in hits] == ["a", "c", "b"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[2].score == pytest.approx(0.0, abs=1e-5)


def test_top_k_truncates():
    idx = InMemoryVectorIndex()
    for i in range(20):
        idx.add(f"c{i}", _unit(1.0, float(i) / 20.0))
    assert len(idx.search([1.0, 0.1], top_k=5)) == 5


def test_min_score_filters():
    idx = InMemoryVectorIndex()
    idx.add("a", _unit(1.0, 0.0))
    idx.add("b", _unit(-1.0, 0.0))  # cosine = -1 with query (1,0)
    hits = idx.search([1.0, 0.0], top_k=10, min_score=0.5)
    assert [h.concept_id for h in hits] == ["a"]


def test_add_replaces_existing_id():
    idx = InMemoryVectorIndex()
    idx.add("a", _unit(1.0, 0.0))
    idx.add("a", _unit(0.0, 1.0))
    assert idx.size() == 1
    hits = idx.search([0.0, 1.0], top_k=1)
    assert hits[0].concept_id == "a"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_remove():
    idx = InMemoryVectorIndex()
    idx.add("a", _unit(1.0, 0.0))
    idx.add("b", _unit(0.0, 1.0))
    idx.add("c", _unit(0.71, 0.71))
    idx.remove("b")
    assert idx.size() == 2
    hits = idx.search([0.0, 1.0], top_k=3)
    ids = {h.concept_id for h in hits}
    assert ids == {"a", "c"}


def test_remove_unknown_is_noop():
    idx = InMemoryVectorIndex()
    idx.add("a", _unit(1.0, 0.0))
    idx.remove("nonexistent")
    assert idx.size() == 1


def test_rebuild_replaces_everything():
    idx = InMemoryVectorIndex()
    idx.add("a", _unit(1.0, 0.0))
    idx.add("b", _unit(0.0, 1.0))
    idx.rebuild([("x", _unit(1.0, 1.0)), ("y", _unit(1.0, -1.0))])
    assert idx.size() == 2
    hits = idx.search(_unit(1.0, 1.0), top_k=2)
    assert hits[0].concept_id == "x"


def test_dim_mismatch_silently_skipped():
    idx = InMemoryVectorIndex()
    idx.add("a", _unit(1.0, 0.0))
    idx.add("b", [0.5, 0.5, 0.5])  # 3-dim, mismatched — skipped
    assert idx.size() == 1


def test_zero_vector_skipped():
    idx = InMemoryVectorIndex()
    idx.add("a", [0.0, 0.0])
    assert idx.size() == 0


def test_returns_unit_normalized_scores():
    """Scores should be in [-1, 1] regardless of input vector magnitude."""
    idx = InMemoryVectorIndex()
    idx.add("a", [10.0, 0.0])  # not unit; index normalizes
    hits = idx.search([0.5, 0.0], top_k=1)
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_with_top_k_larger_than_size():
    idx = InMemoryVectorIndex()
    idx.add("a", _unit(1.0, 0.0))
    idx.add("b", _unit(0.0, 1.0))
    hits = idx.search([1.0, 1.0], top_k=10)
    assert len(hits) == 2


def test_remove_then_re_add_works():
    """The swap-with-last-row deletion strategy should leave the index
    re-usable after a remove."""
    idx = InMemoryVectorIndex()
    idx.add("a", _unit(1.0, 0.0))
    idx.add("b", _unit(0.0, 1.0))
    idx.add("c", _unit(0.5, 0.5))
    idx.remove("b")
    idx.add("d", _unit(-1.0, 0.0))
    assert idx.size() == 3
    # Use min_score=-1.0 to include the antipodal "d" concept too.
    ids_back = {h.concept_id for h in idx.search([1.0, 0.0], top_k=3, min_score=-1.0)}
    assert ids_back == {"a", "c", "d"}
