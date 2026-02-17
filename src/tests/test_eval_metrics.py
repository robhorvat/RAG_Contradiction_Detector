from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_retrieval_metrics_from_rankings,
)


def test_compute_classification_metrics_returns_expected_shape():
    labels = ("Contradictory", "Supporting", "Unrelated")
    y_true = ["Contradictory", "Supporting", "Unrelated", "Supporting"]
    y_pred = ["Contradictory", "Unrelated", "Unrelated", "Supporting"]
    metrics = compute_classification_metrics(y_true, y_pred, labels=labels)

    assert metrics["n_samples"] == 4
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert set(metrics["per_label"].keys()) == set(labels)


def test_compute_retrieval_metrics_from_rankings():
    rankings = [
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
    ]
    relevant_sets = [
        {12},
        {25},
        {30},
    ]
    metrics = compute_retrieval_metrics_from_rankings(rankings, relevant_sets, k_values=(1, 3))
    assert metrics["n_queries"] == 3
    assert metrics["recall_at_k"]["1"] == 0.3333
    assert metrics["recall_at_k"]["3"] == 0.6667
    assert metrics["mrr_at_k"]["1"] == 0.3333
    assert metrics["mrr_at_k"]["3"] == 0.4444
