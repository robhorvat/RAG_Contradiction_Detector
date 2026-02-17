from __future__ import annotations

import json
import re


class OpenAIJSONClient:
    def __init__(self, *, api_key: str, model: str):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "OpenAI support requires the `openai` package. Install with: "
                "`pip install openai`"
            ) from exc
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate_json(self, *, system_prompt: str, user_message: str) -> tuple[dict, str]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        return json.loads(raw), raw


def _extract_json_object(text: str) -> str:
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON object found in model output.")
    return text[first : last + 1]


class GeminiJSONClient:
    def __init__(self, *, api_key: str, model: str):
        try:
            import google.generativeai as genai  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Gemini support requires `google-generativeai`. Install with: "
                "`pip install google-generativeai`"
            ) from exc

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def generate_json(self, *, system_prompt: str, user_message: str) -> tuple[dict, str]:
        prompt = (
            f"{system_prompt.strip()}\n\n"
            f"{user_message.strip()}\n\n"
            "Return ONLY a single JSON object. No markdown. No code fences."
        )
        response = self._model.generate_content(prompt, generation_config={"temperature": 0})
        raw = getattr(response, "text", "") or ""
        return json.loads(_extract_json_object(raw)), raw


class LocalRuleBasedJSONClient:
    """
    Deterministic local fallback mode for no-network demos.
    It is intentionally simple and should be treated as a baseline.
    """

    @staticmethod
    def _extract_context(user_message: str, marker: str) -> str:
        idx = user_message.find(marker)
        if idx == -1:
            return ""
        chunk = user_message[idx + len(marker) :]
        end_idx = chunk.find("**Retrieved Passages")
        if end_idx != -1:
            chunk = chunk[:end_idx]
        return chunk.strip()

    @staticmethod
    def _first_sentence(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return "No specific claim could be extracted from the provided text."
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        return parts[0][:280]

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-zA-Z]{3,}", text.lower())}

    def generate_json(self, *, system_prompt: str, user_message: str) -> tuple[dict, str]:
        del system_prompt
        marker_1 = "**Retrieved Passages from Paper 1"
        marker_2 = "**Retrieved Passages from Paper 2"
        context_1 = self._extract_context(user_message, marker_1)
        context_2 = self._extract_context(user_message, marker_2)

        claim_1 = self._first_sentence(context_1)
        claim_2 = self._first_sentence(context_2)

        tokens_1 = self._token_set(claim_1)
        tokens_2 = self._token_set(claim_2)
        overlap = len(tokens_1 & tokens_2)
        min_size = max(1, min(len(tokens_1), len(tokens_2)))
        overlap_ratio = overlap / min_size

        neg_words = {"no", "not", "without", "none", "lack", "failed"}
        has_neg_1 = any(w in tokens_1 for w in neg_words)
        has_neg_2 = any(w in tokens_2 for w in neg_words)

        if overlap_ratio < 0.10:
            verdict = "Unrelated"
            reason = "Local baseline found low lexical overlap between extracted claims."
        elif has_neg_1 != has_neg_2:
            verdict = "Contradictory"
            reason = "Local baseline detected opposing polarity in overlapping claims."
        else:
            verdict = "Supporting"
            reason = "Local baseline detected aligned polarity with overlapping claim terms."

        payload = {
            "paper_1_claim": claim_1,
            "paper_2_claim": claim_2,
            "analysis": {"verdict": verdict, "justification": reason},
        }
        return payload, json.dumps(payload)
