# Compliance Classifier — Product Thinking

A weekend prototype. The interesting part isn't the app, it's the eval harness
and what it taught me about how this kind of tool should actually be shipped.

## Problem

Compliance analysts spend a lot of time reading onboarding documents looking
for the same handful of patterns: unclear ownership structures, high-risk
countries, suspicious transaction shapes, sanctions exposure. A first-pass
triage tool that surfaces those patterns with evidence quoted from the
document can save real time per case — without trying to replace the analyst.

The bar isn't "be right." The bar is "give the analyst a head start and be
honest about what you're unsure of."

## User

A compliance analyst reviewing 5–20 cases a day. Domain expert, not an AI
expert. Wants outputs they can audit, predictable failure modes, and a clean
way to disagree with the tool.

## Solution

A two-prompt LLM pipeline:

- **Prompt A** classifies the document into one of five categories with a
  confidence signal.
- **Prompt B** extracts red flags as JSON, where every flag must be paired
  with an `evidence` string that appears verbatim in the source document.

Evidence-grounding is the one design choice I'd defend hardest. It stops the
model from inventing flags that aren't in the document, and it gives the
analyst something concrete to point at when explaining a decision.

## What I didn't build, and why

- **No sanctions-list integration.** Real lists have licensing and freshness
  requirements. In production this should be a deterministic name-match step
  that runs **before** the LLM, not inside the prompt.
- **No review UI.** A production version needs a way to capture analyst
  overrides and feed them back. Called out, not faked.
- **No audit log or persistence.** Needed for a regulated deployment,
  straightforward to add, out of scope for a weekend.
- **No model A/B.** Picked one model on purpose. The eval harness makes a
  swap a one-line change.

## Evaluation

15 hand-written test cases across 4 tiers: 5 clear-negative, 5 clear-positive,
3 ambiguous, 2 adversarial. Each case has an expected label, hand-labeled
before any model run.

### Latest run

| Metric | Result |
|--------|--------|
| Overall accuracy | 10/15 = 67% |
| Clear-negative | 5/5 = 100% |
| Clear-positive | 3/5 = 60% |
| Ambiguous | 0/3 = 0% |
| Adversarial | 2/2 = 100% |
| Flag recall | 0.80 |
| Avg latency | 9.25s |
| Cost per 1,000 runs | ~$8.09 (Claude Sonnet) |

### What the tiers tell me

The aggregate number is the least useful thing in the table. The tiers
are where the signal is:

- **100% on clear-negative** is the property I care about most. A triage
  tool that cries wolf on clean cases loses analyst trust fast.
- **60% on clear-positive** — the two misses weren't failures to detect
  risk, they were the model picking the wrong category between KYC and
  AML. That's a category-design problem more than a model problem.
- **0% on ambiguous** is the most useful finding. The model doesn't like
  saying "ambiguous" — when signals point two ways it just picks one.
  The right answer isn't to keep tweaking the prompt. It's to route
  ambiguous-looking inputs to a human **by policy**, not based on the
  model's own confidence.

## What I'd change before production

1. Route ambiguous cases to a human based on input signals, not on the
   model's self-reported confidence.
2. Fix the KYC/AML category overlap — either multi-label or hierarchy.
3. Grow the test set, especially the ambiguous tier.
4. Audit log every inference with its evidence trace.

## Tradeoffs I'd raise with the team

- **Latency.** Two prompts is ~2× a single fused prompt. Worth it for
  being able to eval each step on its own, but I'd revisit if latency
  becomes a user complaint.
- **Cost.** ~$8 per 1,000 runs is fine for low volume, expensive at
  scale. One optimization: skip the flag-extraction prompt when the
  classifier returns "not a compliance concern" — knocks cost down
  meaningfully given the tier distribution.
- **"Ambiguous" can't be doing all the work.** If too many cases end up
  there, the tool isn't earning its keep. I'd track the ambiguous rate
  as a real metric, not an afterthought.
