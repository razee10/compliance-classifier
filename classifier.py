"""Two-step compliance classifier.

Prompt A: classify the document (KYC / AML / sanctions-adjacent / not / ambiguous).
Prompt B: extract red flags with quoted evidence + unknowns the reviewer should check.

The two prompts are kept separate on purpose:
- Cheaper to iterate on each independently.
- Easier to eval each step in isolation.
- Lets us swap the classifier without touching flag extraction.

## Providers

Set `LLM_PROVIDER` in env to pick a backend:
- `anthropic` (default): uses `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`.
- `nvidia`: uses `NVIDIA_API_KEY`, `NVIDIA_MODEL`, `NVIDIA_BASE_URL`. NVIDIA's
  hosted endpoint is OpenAI-compatible, so we talk to it with the `openai` client.

The rest of the pipeline (parsing, evidence grounding, eval harness) is provider-agnostic.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any

# Load `.env` as early as possible so module-level env reads below see it,
# regardless of whether the calling script already invoked `load_dotenv()`.
# `load_dotenv()` is idempotent and safe to call repeatedly.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()

# Per-provider defaults.
# For NVIDIA, we default to a non-reasoning instruct model because reasoning
# models emit long <think>...</think> blocks before the JSON, which blow through
# the token budget and make p95 latency awful. Users can override via NVIDIA_MODEL.
ANTHROPIC_MODEL_DEFAULT = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
NVIDIA_MODEL_DEFAULT = os.environ.get("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Single symbol the rest of the codebase reads. Resolves to the active provider's default.
DEFAULT_MODEL = NVIDIA_MODEL_DEFAULT if DEFAULT_PROVIDER == "nvidia" else ANTHROPIC_MODEL_DEFAULT

# Generous enough to survive a reasoning model's <think> preamble on top of JSON.
MAX_TOKENS = 2048

VALID_LABELS = {
    "KYC-relevant",
    "AML-relevant",
    "sanctions-adjacent",
    "not a compliance concern",
    "ambiguous",
}
VALID_CONFIDENCE = {"low", "medium", "high"}
VALID_SEVERITY = {"info", "warning", "critical"}


CLASSIFY_PROMPT = """You are a compliance-screening assistant reviewing a short text. Your job is to classify it.

Input:
\"\"\"
{document_text}
\"\"\"

Return valid JSON only, matching this schema:
{{
  "label": "KYC-relevant" | "AML-relevant" | "sanctions-adjacent" | "not a compliance concern" | "ambiguous",
  "confidence": "low" | "medium" | "high",
  "primary_reason": "<one short sentence>"
}}

Rules:
- Only classify as "sanctions-adjacent" if the text explicitly mentions a country, entity, or individual commonly on sanctions lists. Do not speculate.
- "KYC-relevant", "AML-relevant", and "sanctions-adjacent" mean there is a CONCERN that needs review. Routine information (verified identity, salary income, stable employment, standard documentation) is NOT a concern -- classify as "not a compliance concern".
- Context-vs-substance rule applies to all categories: if a compliance keyword (high-risk country, sanctions term, AML terminology) appears in the text only as training material, news commentary, or hypothetical example -- not as a counterparty, jurisdiction of operation, or party to the described activity -- do NOT flag the case. Training memos discussing compliance are not themselves compliance concerns.
- Use "ambiguous" when signals point to multiple categories or are weak. "ambiguous" includes cases where a signal exists but lacks context to judge severity. If a case has exactly one soft signal and no corroborating context, prefer "ambiguous" over committing to a category.
- Confidence is "low" if you'd want a human reviewer; "high" only if the signal is unambiguous.
- Return ONLY the JSON object. No prose before or after.
"""

FLAGS_PROMPT = """You are a compliance analyst. Given the same text, list red flags using only evidence from the text itself.

Input:
\"\"\"
{document_text}
\"\"\"

Return valid JSON only:
{{
  "red_flags": [
    {{
      "flag": "<short kebab-case label, e.g. 'high-risk-jurisdiction'>",
      "evidence": "<exact phrase quoted from the input>",
      "severity": "info" | "warning" | "critical"
    }}
  ],
  "unknowns": [
    "<things a human should verify that the text doesn't answer>"
  ]
}}

Rules:
- Never fabricate evidence. Every "evidence" value MUST be a substring that appears verbatim in the input.
- If the text doesn't warrant any flags, return "red_flags": [].
- "unknowns" should list questions the text doesn't answer but a reviewer would ask.
- Return ONLY the JSON object. No prose before or after.
"""


@dataclass
class RedFlag:
    flag: str
    evidence: str
    severity: str
    evidence_grounded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationResult:
    label: str
    confidence: str
    primary_reason: str
    red_flags: list[RedFlag] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)

    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    raw_classify: dict[str, Any] = field(default_factory=dict)
    raw_flags: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["red_flags"] = [f.to_dict() if not isinstance(f, dict) else f for f in self.red_flags]
        return d


import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> dict[str, Any]:
    """Responses sometimes wrap JSON in prose, fences, or reasoning tags. Find the first {...} block."""
    text = text.strip()
    # Reasoning models (DeepSeek-R1, minimax, etc.) emit <think>...</think> before the answer.
    # Strip any such blocks so we don't accidentally parse JSON-looking content inside the reasoning.
    text = _THINK_RE.sub("", text).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------------------
# Provider-specific callers. Each returns (text, input_tokens, output_tokens).
# ---------------------------------------------------------------------------

def _call_anthropic(client: Any, model: str, prompt: str) -> tuple[str, int, int]:
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return (
        response.content[0].text,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )


def _call_openai_compatible(client: Any, model: str, prompt: str) -> tuple[str, int, int]:
    """Works for any OpenAI-compatible endpoint (incl. NVIDIA NIM)."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        # Some NIM models require low temperature to produce strict JSON.
        temperature=0.1,
    )
    text = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
    out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
    return text, in_tok, out_tok


def _build_client(provider: str) -> Any:
    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic()  # picks up ANTHROPIC_API_KEY from env
    if provider == "nvidia":
        from openai import OpenAI
        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError("NVIDIA_API_KEY not set in env. Add it to .env.")
        # NVIDIA's free tier can be flaky; give generous timeout + retries.
        return OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=key,
            timeout=120.0,
            max_retries=3,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Expected 'anthropic' or 'nvidia'.")


def _call_model(
    client: Any, provider: str, model: str, prompt: str
) -> tuple[dict[str, Any], int, int, int]:
    """Returns (parsed_json, latency_ms, input_tokens, output_tokens)."""
    t0 = time.time()
    if provider == "anthropic":
        text, in_tok, out_tok = _call_anthropic(client, model, prompt)
    elif provider == "nvidia":
        text, in_tok, out_tok = _call_openai_compatible(client, model, prompt)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")
    latency_ms = int((time.time() - t0) * 1000)
    parsed = _extract_json(text)
    return parsed, latency_ms, in_tok, out_tok


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(
    document_text: str,
    *,
    client: Any | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> ClassificationResult:
    """Run the two-prompt pipeline against a single document."""
    provider = (provider or DEFAULT_PROVIDER).lower()
    if model is None:
        model = NVIDIA_MODEL_DEFAULT if provider == "nvidia" else ANTHROPIC_MODEL_DEFAULT
    client = client or _build_client(provider)

    result = ClassificationResult(
        label="ambiguous",
        confidence="low",
        primary_reason="",
        provider=provider,
        model=model,
    )

    # --- Prompt A: classification
    try:
        parsed, lat, in_tok, out_tok = _call_model(
            client, provider, model, CLASSIFY_PROMPT.format(document_text=document_text)
        )
        result.raw_classify = parsed
        result.latency_ms += lat
        result.input_tokens += in_tok
        result.output_tokens += out_tok

        label = parsed.get("label", "ambiguous")
        if label not in VALID_LABELS:
            result.errors.append(f"invalid label from model: {label!r}")
            label = "ambiguous"
        confidence = parsed.get("confidence", "low")
        if confidence not in VALID_CONFIDENCE:
            result.errors.append(f"invalid confidence from model: {confidence!r}")
            confidence = "low"

        result.label = label
        result.confidence = confidence
        result.primary_reason = parsed.get("primary_reason", "")
    except Exception as exc:
        result.errors.append(f"classify failed: {exc}")
        return result

    # --- Prompt B: red flags
    try:
        parsed, lat, in_tok, out_tok = _call_model(
            client, provider, model, FLAGS_PROMPT.format(document_text=document_text)
        )
        result.raw_flags = parsed
        result.latency_ms += lat
        result.input_tokens += in_tok
        result.output_tokens += out_tok

        for raw in parsed.get("red_flags", []):
            evidence = raw.get("evidence", "")
            severity = raw.get("severity", "info")
            if severity not in VALID_SEVERITY:
                result.errors.append(f"invalid severity: {severity!r}")
                severity = "info"
            grounded = bool(evidence) and evidence in document_text
            if not grounded:
                result.errors.append(
                    f"ungrounded evidence: {evidence!r} not found in input (flag dropped from trusted view)"
                )
            result.red_flags.append(
                RedFlag(
                    flag=raw.get("flag", "unknown"),
                    evidence=evidence,
                    severity=severity,
                    evidence_grounded=grounded,
                )
            )
        result.unknowns = list(parsed.get("unknowns", []))
    except Exception as exc:
        result.errors.append(f"flag extraction failed: {exc}")

    return result


if __name__ == "__main__":
    # Quick smoke test. Requires the active provider's API key in env.
    sample = (
        "Company Alpha Trading SA. Registered in Panama. Five directors, three listed as "
        "residents of jurisdictions flagged by FATF. UBO structure unclear -- two intermediate "
        "holding entities in the BVI."
    )
    out = classify(sample)
    print(json.dumps(out.to_dict(), indent=2, default=str))
