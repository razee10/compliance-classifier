"""Eval harness for the compliance classifier.

Runs every case in `test-cases.json` through the classifier, computes:
- classification accuracy
- average flag recall (overlap between expected and predicted flag sets)
- average latency
- estimated cost (token-based, using the model's published pricing)

Writes a human-readable Markdown report to `eval/results/run-YYYY-MM-DD-HHMM.md`.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --cases eval/test-cases.json --out eval/results

Requires ANTHROPIC_API_KEY in env.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow running this file directly: `python eval/run_eval.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from classifier import DEFAULT_MODEL, DEFAULT_PROVIDER, _build_client, classify

load_dotenv()

# Approximate prices per 1M tokens (USD). Update if model or pricing changes.
# NVIDIA's NIM free tier is $0 per token within rate limits.
MODEL_PRICING = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-opus-4-7": {"input": 15.0, "output": 75.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
}


@dataclass
class CaseResult:
    case_id: str
    tier: str
    expected_label: str
    got_label: str
    label_correct: bool
    expected_flags: list[str]
    got_flags: list[str]
    flag_recall: float
    confidence: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    primary_reason: str
    errors: list[str]


def flag_recall(expected: list[str], got: list[str]) -> float:
    """Recall of expected flags. Lenient string match (case-insensitive substring either way)."""
    if not expected:
        return 1.0  # nothing to recall
    got_norm = [g.lower() for g in got]
    hits = 0
    for exp in expected:
        e = exp.lower()
        if any(e in g or g in e for g in got_norm):
            hits += 1
    return hits / len(expected)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]


def run(cases_path: Path, out_dir: Path, provider: str, model: str) -> Path:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    client = _build_client(provider)

    results: list[CaseResult] = []
    t_start = time.time()

    for case in cases:
        print(f"[{case['id']}] tier={case['tier']} ...", flush=True)
        out = classify(case["input"], client=client, provider=provider, model=model)
        got_flags = [f.flag for f in out.red_flags]
        recall = flag_recall(case.get("expected_flags", []), got_flags)
        results.append(
            CaseResult(
                case_id=case["id"],
                tier=case.get("tier", "unknown"),
                expected_label=case["expected_label"],
                got_label=out.label,
                label_correct=out.label == case["expected_label"],
                expected_flags=case.get("expected_flags", []),
                got_flags=got_flags,
                flag_recall=recall,
                confidence=out.confidence,
                latency_ms=out.latency_ms,
                input_tokens=out.input_tokens,
                output_tokens=out.output_tokens,
                primary_reason=out.primary_reason,
                errors=out.errors,
            )
        )

    total_wall = time.time() - t_start

    # --- aggregate metrics
    n = len(results)
    correct = sum(r.label_correct for r in results)
    accuracy = correct / n if n else 0
    avg_recall = sum(r.flag_recall for r in results) / n if n else 0
    avg_latency = sum(r.latency_ms for r in results) / n if n else 0
    total_input = sum(r.input_tokens for r in results)
    total_output = sum(r.output_tokens for r in results)
    total_cost = estimate_cost(model, total_input, total_output)
    cost_per_1k_runs = (total_cost / n) * 1000 if n else 0

    # --- write report
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
    report_path = out_dir / f"run-{stamp}.md"

    misses = [r for r in results if not r.label_correct or r.flag_recall < 1.0]
    by_tier: dict[str, list[CaseResult]] = {}
    for r in results:
        by_tier.setdefault(r.tier, []).append(r)

    lines: list[str] = []
    lines.append(f"# Eval Run -- {stamp}")
    lines.append("")
    lines.append(f"**Provider:** `{provider}`")
    lines.append(f"**Model:** `{model}`")
    lines.append(f"**Test cases:** {n}")
    lines.append(f"**Classification accuracy:** {correct}/{n} = {accuracy:.0%}")
    lines.append(f"**Average flag recall:** {avg_recall:.2f}")
    lines.append(f"**Average latency per case:** {avg_latency/1000:.2f}s")
    lines.append(f"**Total wall time:** {total_wall:.1f}s")
    lines.append(f"**Tokens:** {total_input:,} in / {total_output:,} out")
    lines.append(f"**Estimated cost per 1,000 runs:** ${cost_per_1k_runs:.2f}")
    lines.append("")

    lines.append("## Accuracy by tier")
    lines.append("")
    lines.append("| Tier | Cases | Correct | Accuracy |")
    lines.append("|------|-------|---------|----------|")
    for tier, rs in by_tier.items():
        c = sum(r.label_correct for r in rs)
        lines.append(f"| {tier} | {len(rs)} | {c} | {c/len(rs):.0%} |")
    lines.append("")

    lines.append("## Misses & partial misses")
    lines.append("")
    if not misses:
        lines.append("_None -- all cases matched expected label and flags._")
    else:
        lines.append("| Case | Tier | Expected | Got | Confidence | Flag recall |")
        lines.append("|------|------|----------|-----|------------|-------------|")
        for r in misses:
            lines.append(
                f"| {r.case_id} | {r.tier} | {r.expected_label} | {r.got_label} | "
                f"{r.confidence} | {r.flag_recall:.2f} |"
            )
        lines.append("")
        lines.append("### Per-miss notes")
        lines.append("")
        for r in misses:
            lines.append(f"- **{r.case_id}** ({r.tier})")
            lines.append(f"  - expected label: `{r.expected_label}` -- got: `{r.got_label}`")
            lines.append(f"  - expected flags: {r.expected_flags}")
            lines.append(f"  - got flags: {r.got_flags}")
            lines.append(f"  - model's reason: {r.primary_reason}")
            if r.errors:
                lines.append(f"  - pipeline errors: {r.errors}")
            lines.append("")

    lines.append("## All cases (raw)")
    lines.append("")
    lines.append("| Case | Tier | Expected | Got | Conf | Recall | Latency |")
    lines.append("|------|------|----------|-----|------|--------|---------|")
    for r in results:
        ok = ":white_check_mark:" if r.label_correct else ":x:"
        lines.append(
            f"| {r.case_id} | {r.tier} | {r.expected_label} | {ok} {r.got_label} | "
            f"{r.confidence} | {r.flag_recall:.2f} | {r.latency_ms} ms |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Flag recall uses a lenient case-insensitive substring match because the model is free to "
        "label flags however it likes (e.g. `high-risk-jurisdiction` vs `high_risk_country`). "
        "A stricter matcher would lower recall but reward consistent labels."
    )
    lines.append(
        "- Cost is estimated from token counts and the model's published per-token price; treat as "
        "order of magnitude, not contract billing."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Also dump the raw data as JSON for downstream analysis.
    raw_path = out_dir / f"run-{stamp}.json"
    raw_path.write_text(
        json.dumps([r.__dict__ for r in results], indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print(f"Wrote {report_path}")
    print(f"Wrote {raw_path}")
    print(
        f"Accuracy {accuracy:.0%}  |  flag recall {avg_recall:.2f}  |  "
        f"avg latency {avg_latency/1000:.2f}s  |  ~${cost_per_1k_runs:.2f}/1k runs"
    )
    return report_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=Path, default=ROOT / "eval" / "test-cases.json")
    p.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    p.add_argument("--provider", default=DEFAULT_PROVIDER, help="anthropic | nvidia")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    required_key = "NVIDIA_API_KEY" if args.provider == "nvidia" else "ANTHROPIC_API_KEY"
    if not os.environ.get(required_key):
        print(
            f"ERROR: {required_key} not set in env. Add it to .env and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    run(args.cases, args.out, args.provider, args.model)


if __name__ == "__main__":
    main()
