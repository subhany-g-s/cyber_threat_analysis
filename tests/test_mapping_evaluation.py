import json

import pytest

from src.mapping.evaluation import evaluate_mapping, load_examples


class FakeCandidate:
    def __init__(self, technique_id, similarity_score):
        self.technique_id = technique_id
        self.similarity_score = similarity_score


class FakeMapper:
    """Deterministic mapper for testing evaluator metrics without SBERT."""

    top_k = 3

    def __init__(self, rankings):
        self.rankings = rankings

    def embed_texts(self, texts):
        # Each input's position identifies its predetermined ranking.
        return list(range(len(texts)))

    def rank_vector(self, vector):
        return [
            FakeCandidate(technique_id, score)
            for technique_id, score in self.rankings[vector]
        ]


def test_load_examples_accepts_valid_file(tmp_path):
    path = tmp_path / "examples.json"
    path.write_text(json.dumps({
        "examples": [
            {"id": "positive", "text": "PowerShell execution",
             "expected_technique_ids": ["T1059.001"]},
            {"id": "negative", "text": "Routine maintenance",
             "expected_technique_ids": []}
        ]
    }), encoding="utf-8")

    examples = load_examples(str(path))
    assert len(examples) == 2


@pytest.mark.parametrize("payload", [
    {"examples": []},
    {"examples": [{"text": "missing labels"}]},
    {"examples": [{"expected_technique_ids": ["T1059.001"]}]},
])
def test_load_examples_rejects_invalid_payload(tmp_path, payload):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_examples(str(path))


def test_evaluation_measures_ranking_and_thresholds():
    examples = [
        {"id": "positive", "text": "PowerShell execution",
         "expected_technique_ids": ["T1059.001"]},
        {"id": "negative", "text": "Routine maintenance",
         "expected_technique_ids": []},
    ]
    mapper = FakeMapper([
        [("T1059.001", 0.80), ("T1059", 0.70)],
        [("T1561.001", 0.40), ("T1561", 0.30)],
    ])

    result = evaluate_mapping(examples, mapper)

    assert result["top_k_hit_rate"]["top1"] == 1.0
    assert result["mrr"] == 1.0

    at_045 = next(
        row for row in result["threshold_table"]
        if row["threshold"] == 0.45
    )
    assert at_045["predicted"] == 1
    assert at_045["precision_top1"] == 1.0
    assert at_045["recall_top1"] == 1.0
    assert at_045["negatives_left_unmatched"] == 1.0


def test_evaluation_reports_no_threshold_when_no_top1_hits():
    examples = [
        {"text": "PowerShell execution",
         "expected_technique_ids": ["T1059.001"]},
    ]
    mapper = FakeMapper([
        [("T1134.004", 0.80), ("T1059.001", 0.70)],
    ])

    result = evaluate_mapping(examples, mapper)

    assert result["top_k_hit_rate"]["top1"] == 0.0
    assert result["top_k_hit_rate"]["top3"] == 1.0
    assert result["suggested_candidate_threshold"] is None
