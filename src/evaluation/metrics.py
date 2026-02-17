from __future__ import annotations

from collections.abc import Iterable


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    *,
    labels: tuple[str, ...],
) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not labels:
        raise ValueError("labels must not be empty")

    support_by_label = {label: 0 for label in labels}
    tp = {label: 0 for label in labels}
    fp = {label: 0 for label in labels}
    fn = {label: 0 for label in labels}

    for truth, pred in zip(y_true, y_pred):
        if truth in support_by_label:
            support_by_label[truth] += 1
        for label in labels:
            if truth == label and pred == label:
                tp[label] += 1
            elif truth != label and pred == label:
                fp[label] += 1
            elif truth == label and pred != label:
                fn[label] += 1

    per_label = {}
    f1_values: list[float] = []
    for label in labels:
        precision = _safe_div(tp[label], tp[label] + fp[label])
        recall = _safe_div(tp[label], tp[label] + fn[label])
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(support_by_label[label]),
        }
        f1_values.append(float(f1))

    accuracy = _safe_div(sum(int(t == p) for t, p in zip(y_true, y_pred)), len(y_true))
    macro_f1 = _safe_div(sum(f1_values), len(labels))

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_label": per_label,
        "n_samples": int(len(y_true)),
    }


def compute_retrieval_metrics_from_rankings(
    rankings: Iterable[list[int]],
    relevant_sets: Iterable[set[int]],
    *,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    k_values = tuple(sorted({int(k) for k in k_values if int(k) > 0}))
    if not k_values:
        raise ValueError("k_values must contain at least one positive integer")

    recall_hits = {k: 0 for k in k_values}
    mrr_sum = {k: 0.0 for k in k_values}
    n_queries = 0

    for ranking, relevant in zip(rankings, relevant_sets):
        if not relevant:
            continue
        n_queries += 1
        first_rank = None
        for rank_idx, doc_id in enumerate(ranking, start=1):
            if doc_id in relevant:
                first_rank = rank_idx
                break

        for k in k_values:
            topk = ranking[:k]
            if any(doc_id in relevant for doc_id in topk):
                recall_hits[k] += 1
            if first_rank is not None and first_rank <= k:
                mrr_sum[k] += 1.0 / float(first_rank)

    recall_at_k = {str(k): round(_safe_div(recall_hits[k], n_queries), 4) for k in k_values}
    mrr_at_k = {str(k): round(_safe_div(mrr_sum[k], n_queries), 4) for k in k_values}

    return {
        "n_queries": int(n_queries),
        "recall_at_k": recall_at_k,
        "mrr_at_k": mrr_at_k,
    }
