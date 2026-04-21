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

Hand-built test set of 15 cases across 4 difficulty tiers: 5 clear-negative,
5 clear-positive, 3 ambiguous, 2 adversarial. Each case carries an expected
label and an expected flag set, hand-labeled before any model run.

### Headline numbers (latest run)

| Metric | Result |
|--------|--------|
| Overall label accuracy | **10/15 = 67%** |
| Clear-negative tier | 5/5 = 100% |
| Clear-positive tier | 3/5 = 60% |
| Ambiguous tier | 0/3 = 0% |
| Adversarial tier | 2/2 = 100% |
| Avg latency per case | ~9.25s |
| Estimated cost per 1,000 runs | ~$8.09 (Claude Sonnet) |

### What the tier breakdown actually tells me

The aggregate number (67%) is the least interesting thing in this table.
The tier-level breakdown is where the product signal is:

- **Clear-negative at 100%** — the model is not trigger-happy on benign
  text. For a compliance triage tool, this is the single most important
  property: false positives burn analyst time and erode trust faster than
  misses do.

- **Adversarial at 100%** — including the case where a benign news
  reference mentions a sanctioned country. The tightened prompt rule
  ("don't flag if a country appears only as news/context") held.

- **Clear-positive at 60%** — two misses (case-08, case-10) were
  AML-vs-KYC category confusion, not a failure to detect risk. The model
  saw the red flags; it picked the adjacent category. This is a taxonomy
  problem more than a model problem — the categories overlap in reality,
  and a production version would likely either (a) allow multi-label
  output or (b) structure them hierarchically (KYC-relevant as the
  parent, AML-relevant as a child when specific AML signals are present).

- **Ambiguous tier at 0%** — the most interesting failure mode. The model
  does not like saying "ambiguous." When signals point two ways, it
  picks the stronger-feeling one and commits. This is a well-known LLM
  behavior (miscalibrated confidence under ambiguity) and it is *itself
  a product finding*: the right response to this is not to keep
  re-prompting until the model gets it right, it's to **route ambiguous
  cases to a human reviewer by policy**, not by model self-assessment.
  The eval just told me exactly which kinds of inputs need the HITL
  path.

### What I'd change before production

1. **HITL routing at the input layer, not just at low `confidence`.**
   Don't trust the model to self-report ambiguity. Add an independent
   signal (e.g., short-text + multiple category keywords + missing UBO
   info = route to human regardless of model output).

2. **Multi-label or hierarchical categories.** AML-vs-KYC confusion is
   a category design issue. In production, I'd either let a case be
   tagged both or define KYC as the umbrella and AML as a specialization
   triggered by specific patterns (structuring, opaque UBO, high-risk
   jurisdictions).

3. **Expand the ambiguous tier in the test set.** 3 cases is too few to
   know if a prompt change helps. Before shipping I'd grow this to 15+
   and re-run weekly.

4. **Log every inference with evidence trace.** Already enforced at
   prompt level (evidence strings must be substrings of input); in prod
   I'd verify this at parse time and reject outputs that fail the check.

## Governance considerations (what I'd do for a regulated deployment)

- **Audit log** — every inference stored with input, output, model
  version, prompt version, timestamp, reviewer ID if routed.
- **Evidence-grounded output** — enforced at prompt level today; in
  production, verified at parse time.
- **Human-in-the-loop by policy** — ambiguous-tier routing driven by
  input signals, not by the model's self-reported `confidence`.
- **Rolling re-evaluation** — weekly eval against a growing test set,
  alert on any tier dropping more than 10 points run-over-run.
- **Explicit scope boundaries** — the tool outputs classifications and
  evidence-quoted flags. It does not give advice. It does not auto-close
  or auto-escalate cases.

## Metrics I'd track in production

| Metric | Why it matters |
|--------|----------------|
| Per-tier precision / recall (not just aggregate accuracy) | Aggregate hides the 0% ambiguous tier |
| Analyst time saved per case | Product value |
| % of cases routed to HITL | Routing load, unit economics |
| Analyst override rate (model said X, human changed to Y) | Trust + drift signal |
| Cost per case | Unit economics; important if moving off free inference tier |

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
