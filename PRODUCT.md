# Compliance Classifier — Product Thinking

A short PM write-up for a deliberately narrow weekend prototype. The point is not the artifact — it's the way the artifact is reasoned about.

## Problem

Compliance analysts at fintechs and banks spend hours reading onboarding documents, transaction summaries, and KYC extracts looking for a small set of recurring patterns: high-risk jurisdictions, unclear UBO chains, structuring, sanctions exposure, PEP hits. A first-pass triage that surfaces those patterns — with **evidence quoted from the source document** — can save 30–60 minutes per case and let humans spend their time on the cases that actually need judgment.

The bar isn't "replace the analyst." The bar is "give the analyst a 90-second head start on every case."

## User

A compliance analyst at a mid-sized fintech, reviewing 5–20 cases per day. Domain expert, not an AI expert. Doesn't want to read another vendor pitch about "AI-powered compliance." Wants three things from any tool that lands on their desk:

1. **Reasons they can audit.** No black-box scores.
2. **Predictable failure modes.** They need to know what the tool gets wrong, not just what it gets right.
3. **A clean way to disagree.** Every output should be a starting point, not a verdict.

## Solution sketch

A two-prompt LLM pipeline:

- **Prompt A** classifies the document into one of five categories with a confidence signal.
- **Prompt B** extracts red flags as structured JSON, where every flag is paired with an `evidence` string that **must** appear verbatim in the source. The pipeline rejects ungrounded evidence at parse time.

Both prompts return strict JSON. The UI surfaces label, confidence, primary reason, flags (with severity + evidence), unknowns the reviewer should verify, and a raw-output drawer for audit.

Evidence-grounding is the most important design choice. It removes the single biggest hallucination vector for this use case ("the model invented a flag that isn't in the document") and gives the analyst something concrete to point at when explaining the decision.

## What I deliberately did not build (and why)

- **No real sanctions-list integration (OFAC / EU consolidated).** Licensing matters, freshness matters even more, and a prototype shouldn't lean on lists I can't redistribute. In production, the sanctions check would be a deterministic name-match step that runs **before** the LLM, not as part of it.
- **No model A/B benchmark.** A PM prototype should pick one model intentionally and evaluate it well, not benchmark four models superficially. The eval harness makes a future swap a one-flag change.
- **No human-in-the-loop UI (review queue, override capture, training-data export).** That would be the first thing I'd build next; here it's called out as required-for-prod, not faked.
- **No persistence / audit log.** A regulated deployment would need every prediction stored with input, output, model version, prompt version, and timestamp. Out of scope for one weekend; trivial to add.
- **No fine-tuning.** Premature. The prompt is short and the failure modes I observed are mostly addressable with a stricter rule or a one-shot example, not training data.

## Evaluation

Hand-built test set of **15 cases across four tiers**:

| Tier | Cases | What it tests |
|------|-------|---------------|
| Clear negative | 5 | False-positive rate on plain commercial activity |
| Clear positive | 5 | Catches obvious KYC / AML / sanctions signals |
| Ambiguous | 3 | Should return `ambiguous` or low confidence — not over-confident |
| Adversarial | 2 | News context that name-drops a sanctioned country, or training material that lists high-risk jurisdictions as examples |

The harness (`eval/run_eval.py`) computes classification accuracy, average flag recall, average latency, and an estimated cost per 1,000 runs. Per-miss breakdown lists the model's own stated reason — useful when deciding whether the fix is "tighter prompt rule" or "this category is genuinely ambiguous."

### Current numbers (run 2026-04-20, `meta/llama-3.3-70b-instruct` via NVIDIA NIM free tier)

| Metric | Value |
|--------|-------|
| Classification accuracy | **8 / 15 = 53%** |
| Average flag recall | **0.80** |
| Average latency per case | **3.4 s** |
| Total wall time for 15 cases | 51 s |
| Tokens used | 8,250 in / 2,201 out |
| Cost per 1,000 runs | **$0** (free tier); ~$0.15 equivalent on Anthropic Sonnet |

Accuracy by tier:

| Tier | Cases | Correct |
|------|-------|---------|
| Clear negative | 5 | 4 (80%) |
| Clear positive | 5 | 3 (60%) — the two misses flipped KYC↔AML |
| Ambiguous | 3 | 0 (0%) |
| Adversarial | 2 | 1 (50%) |

### What the eval taught me

The 53% headline number is less interesting than the shape of the misses. Three patterns:

**1. The classifier won't call anything "ambiguous."** Every ambiguous-tier case got a confident specific label. The prompt says "use ambiguous when signals point to multiple categories or are weak," but in practice the model picks whichever signal looks strongest and commits. That's a *prompt-level* bug, not a model-capability one — I'd fix it by adding an explicit tie-breaker rule ("if two or more labels plausibly fit, prefer `ambiguous` with `medium` confidence") and by showing one `ambiguous` example in-prompt.

**2. KYC vs AML is a genuine category-design question, not a classification error.** Case-08 (unsupported source of wealth + shell companies) and case-10 (PEP + large initial deposit + vague source of funds) both got flipped from KYC to AML. Both can plausibly be either, depending on whether you care more about *who the customer is* or *what the money looks like*. Before tightening the prompt, I'd go back to analysts and ask: do you want a category for "onboarding concerns" and a separate one for "transaction concerns," or is a single "customer-risk" category more useful? The eval caught this because the test set was hand-written; a CI-only eval using analyst labels from production would have surfaced it even faster.

**3. Adversarial case-15 failed in exactly the way the plan predicted.** The input is an internal training memo that mentions "AML obligations under the EU's 5MLD" and names fictional companies in North Korea and Syria as *examples*. The model classified it `AML-relevant` with high confidence. The fix is a rule the production version absolutely needs: "if the text is a training document, policy excerpt, or news article that references sanctioned jurisdictions without any counterparty exposure, classify as `not a compliance concern`." This is the kind of subtle failure that only surfaces with adversarial tests — and why I think the test set is the most important artifact in this repo.

### What I'd change in the next prompt iteration

In priority order, smallest first:

1. Add an explicit "prefer ambiguous in a tie" rule to Prompt A, plus one in-prompt example.
2. Constrain the flag vocabulary to a fixed taxonomy per category. This would turn the current soft substring-match recall (0.80) into a precision/recall split that's easier to reason about and easier for an analyst to trust.
3. Add a rule for meta-documents (training, policy, news). This is the adversarial-case fix.
4. Only after those are stable: swap to a stronger model (Claude Sonnet) and re-run. The point isn't to maximize accuracy with bigger models; it's to drive accuracy up with *prompt* changes first, because those are the cheap, portable wins.

### Tradeoffs this run exposed

- **Model choice is the infrastructure decision.** The first NVIDIA run used `minimaxai/minimax-m2.7`, a reasoning model that emits `<think>` blocks before the JSON. It blew through the token budget, produced parse failures, and averaged 48s per case with frequent connection errors. Swapping to `meta/llama-3.3-70b-instruct` (a non-reasoning instruct model) took average latency from 48s → 3.4s with no prompt change. For a regulated-industry production deployment this matters: reasoning models are currently too unpredictable for the latency SLAs I'd expect.
- **Free tier is demo-viable, not production-viable.** NVIDIA NIM free tier cost us $0 but I observed intermittent connection resets under even mild load. Production would need either a paid NVIDIA tier, a self-hosted NIM, or a provider with an SLA (Anthropic / OpenAI / Bedrock).
- **Lenient flag matching is doing real work.** Average recall 0.80 relies on substring-matching `unclear-ubo` ≈ `unclear-ubo-structure`. A strict matcher would drop recall into the 0.3–0.5 range and would surface the vocab-drift problem as a first-class metric. I'd prefer the strict version in production — it's ugly-looking but actionable.

Full per-run reports (including the per-miss model reasoning) are in `eval/results/`.

## Governance considerations (for a regulated deployment)

| Control | How it's met |
|---------|--------------|
| **Audit log** | Every inference stored with input, output, model+prompt versions, timestamp. Not built here; would be a row-per-call append-only table. |
| **Evidence-grounded output** | Already enforced at prompt level *and* verified post-hoc — flag is marked ungrounded if its `evidence` string isn't a substring of the input. |
| **Human-in-the-loop at low confidence** | Anything at `confidence: "low"` routes to a reviewer; never auto-actioned. The UI already warns on low confidence. |
| **Monthly re-evaluation** | Test set lives in version control; CI re-runs the eval whenever the prompt or model changes; regressions block merge. |
| **Out-of-scope guardrails** | The model never outputs advice or recommendations — only classifications and flagged evidence. The system prompt is explicit about this. |
| **Right to explanation** | Every output ships with a `primary_reason`, evidence quotes, and an `unknowns` list. There is no opaque score. |

## Metrics I'd track in production

| Metric | Why it matters |
|--------|----------------|
| Per-case precision / recall by category | Correctness — and an early-warning signal for prompt drift |
| Analyst time-to-decision per case | Direct product value |
| % of cases flagged as `low confidence` | Routing load on senior reviewers |
| Inter-rater agreement (analyst vs. model) | Trust signal — if it dips, retrain the prompt or add few-shot examples |
| Cost per case (input + output tokens) | Unit economics; informs model-tier decisions |
| Override rate (analyst disagrees with label) | Highest-leverage learning signal |

## What I'd build next

1. **Multi-jurisdiction awareness.** Country-specific red-flag rules (FATF grey/black list, EU high-risk-third-country list) injected as structured context, not freeform prompt text.
2. **Per-tenant red-flag taxonomies.** Each customer can extend or restrict the flag vocabulary; eval set extends accordingly.
3. **Sanctions-list integration as a deterministic pre-step.** Name match against OFAC / EU consolidated runs first, then the LLM gets the result as input — never the other way around.
4. **Active learning loop.** Cases the analyst overrides feed back into the test set; a weekly job surfaces new failure patterns.
5. **Prompt versioning + canary.** New prompt versions deploy to 10% of traffic, eval'd against live overrides for two weeks before full rollout.

## Tradeoffs I'd flag to the team early

- **Latency vs. accuracy.** Running both prompts is ~2× the latency of a single fused prompt. Worth it for the per-step evaluability, but I'd revisit if p95 latency becomes a problem.
- **Lenient flag matching in eval.** The current matcher uses substring overlap because the model is free to label flags however it wants. A stricter matcher would reward consistent vocabulary but hide real-but-differently-named hits. The right answer is probably to constrain the flag vocabulary in the prompt itself once we have enough analyst feedback to know what the canonical set should be.
- **"Ambiguous" is doing a lot of work.** It's the right escape hatch for a triage tool, but if too many cases land there, the tool isn't earning its keep. I'd track the ambiguous-rate as a first-class metric and tune the prompt down (or up) based on analyst feedback about whether those cases are truly hard.
