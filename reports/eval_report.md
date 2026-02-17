# Evaluation Report

- Generated at (UTC): `2026-02-17T19:47:10+00:00`
- Verdict pairs evaluated: `638`

## Retrieval Metrics (Lexical Baseline)
- Queries evaluated: `188`
- Recall@k:
  - k=1: `0.3989`
  - k=3: `0.5213`
  - k=5: `0.5479`
  - k=10: `0.6064`
- MRR@k:
  - k=1: `0.3989`
  - k=3: `0.4548`
  - k=5: `0.4604`
  - k=10: `0.4679`

## Verdict Metrics

| Model | Accuracy | Macro-F1 | Contradiction-F1 |
|---|---:|---:|---:|
| majority_unrelated | 0.4702 | 0.2132 | 0.0 |
| heuristic | 0.5784 | 0.4925 | 0.2045 |
| torch_verifier | 0.4843 | 0.2452 | 0.0 |

## Quality Gate
- Status: `fail`
- Candidate model: `torch_verifier`
- macro_f1_min: actual `0.2452` vs threshold `0.7` => `FAIL`
- delta_over_heuristic_min: actual `-0.2473` vs threshold `0.1` => `FAIL`
- Reason: `evaluated torch verifier against thresholds`
