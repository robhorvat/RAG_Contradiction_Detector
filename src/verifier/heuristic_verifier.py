from __future__ import annotations

import re


class HeuristicContradictionVerifier:
    """
    Lightweight lexical baseline for contradiction/support/unrelated verdicts.
    This is intentionally simple and deterministic for reproducible benchmarking.
    """

    NEGATION_TOKENS = {
        "no",
        "not",
        "never",
        "none",
        "without",
        "lack",
        "lacks",
        "failed",
        "fails",
        "absence",
    }

    STOPWORDS = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "were",
        "was",
        "are",
        "has",
        "have",
        "had",
        "into",
        "after",
        "before",
        "among",
        "between",
        "than",
        "their",
        "them",
        "also",
        "show",
        "shows",
        "study",
        "results",
    }

    def __init__(self, min_overlap_ratio: float = 0.12):
        self.min_overlap_ratio = float(min_overlap_ratio)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = {tok for tok in re.findall(r"[a-zA-Z]{3,}", (text or "").lower())}
        return tokens

    def _content_tokens(self, text: str) -> set[str]:
        tokens = self._tokenize(text)
        return {tok for tok in tokens if tok not in self.STOPWORDS}

    def predict(self, claim_1: str, claim_2: str) -> dict:
        tokens_1 = self._content_tokens(claim_1)
        tokens_2 = self._content_tokens(claim_2)

        if not tokens_1 or not tokens_2:
            return {
                "verdict": "Unrelated",
                "confidence": 0.20,
                "overlap_ratio": 0.0,
                "reason": "One or both claims are too short for lexical baseline analysis.",
            }

        overlap = tokens_1 & tokens_2
        denom = max(1, min(len(tokens_1), len(tokens_2)))
        overlap_ratio = len(overlap) / denom

        has_neg_1 = any(tok in self.NEGATION_TOKENS for tok in tokens_1)
        has_neg_2 = any(tok in self.NEGATION_TOKENS for tok in tokens_2)
        opposite_polarity = has_neg_1 != has_neg_2

        if overlap_ratio < self.min_overlap_ratio:
            verdict = "Unrelated"
            confidence = min(0.95, 0.55 + (self.min_overlap_ratio - overlap_ratio))
            reason = "Low lexical overlap between extracted claims."
        elif opposite_polarity:
            verdict = "Contradictory"
            confidence = min(0.98, 0.60 + overlap_ratio)
            reason = "Shared topic terms with opposite negation polarity."
        else:
            verdict = "Supporting"
            confidence = min(0.95, 0.50 + overlap_ratio)
            reason = "Shared topic terms with aligned polarity."

        return {
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "overlap_ratio": round(float(overlap_ratio), 4),
            "reason": reason,
            "shared_terms": sorted(overlap),
        }
