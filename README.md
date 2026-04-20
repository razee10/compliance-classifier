# Compliance Classifier

A small applied-AI prototype that triages short compliance-adjacent text and surfaces red flags with quoted evidence. Built as a weekend prototype for an AI-PM application; intentionally narrow in scope and explicit about what it would need to be production-ready.

> See [`PRODUCT.md`](PRODUCT.md) for the product-thinking write-up: problem framing, what was deliberately left out, governance, metrics, and what I'd build next.

## What it does

Given a short text — a company profile, transaction summary, or KYC extract — the app:

1. **Classifies** it into one of: `KYC-relevant` / `AML-relevant` / `sanctions-adjacent` / `not a compliance concern` / `ambiguous`, with a `low` / `medium` / `high` confidence signal.
2. **Extracts red flags** with **evidence quoted verbatim from the input** (substring-checked at parse time — flags whose evidence isn't in the source are flagged as ungrounded).
3. **Lists "unknowns"** the model thinks a human reviewer should follow up on.

The app is one screen, with a few canned examples in the sidebar.

## Architecture (and why)

A two-prompt pipeline, not a single prompt:

| Step | Prompt | Returns |
|------|--------|---------|
| A | Classify | `label`, `confidence`, `primary_reason` |
| B | Extract red flags | `red_flags[]`, `unknowns[]` |

**Why split it:**
- Each step can be evaluated independently — classifier accuracy and flag recall are separate metrics, separate failure modes, separate fix paths.
- Each prompt is shorter and cheaper, and you can iterate on one without re-testing the other.
- Evidence-grounding logic only matters for step B, so it doesn't pollute step A.

**Hallucination guardrail:** every red-flag's `evidence` field must be a verbatim substring of the input. Enforced post-hoc in `classifier.py`; if the model fabricates, the flag is marked `evidence_grounded: false` and surfaced as suspect in the UI.

## Project layout

```
.
├── app.py                      # Streamlit UI
├── classifier.py               # Two-prompt pipeline + evidence grounding check
├── requirements.txt
├── .env.example                # Copy to .env, add your provider key (Anthropic or NVIDIA)
├── PRODUCT.md                  # PM write-up
└── eval/
    ├── test-cases.json         # 15 hand-written cases across 4 difficulty tiers
    ├── run_eval.py             # Runs all cases, computes accuracy/recall/cost
    └── results/                # Markdown + JSON eval reports per run
```

## Setup

Requires Python 3.10+ and an API key from one of the supported providers.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your provider + key
cp .env.example .env
# then edit .env (see "Providers" below)

# 3. Run the app
streamlit run app.py
```

The app opens at <http://localhost:8501>. Pick an example from the sidebar or paste your own text.

### Providers

The classifier is provider-agnostic. Pick one of two backends via the `LLM_PROVIDER` env var:

| Provider | Env vars | Notes |
|----------|----------|-------|
| `anthropic` (default) | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` (default `claude-sonnet-4-6`) | Paid API; get a key at [console.anthropic.com](https://console.anthropic.com/). |
| `nvidia` | `NVIDIA_API_KEY`, optional `NVIDIA_MODEL` (default `meta/llama-3.3-70b-instruct`), `NVIDIA_BASE_URL` | OpenAI-compatible endpoint. Free-tier credits at [build.nvidia.com](https://build.nvidia.com/). |

`.env.example` has both blocks commented-in. To use NVIDIA:

```dotenv
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...
NVIDIA_MODEL=meta/llama-3.3-70b-instruct
```

The eval harness reads the same env vars, but you can override per run:

```bash
python eval/run_eval.py --provider nvidia --model meta/llama-3.3-70b-instruct
```

## Running the eval

```bash
python eval/run_eval.py
# or pin a specific backend for this run:
python eval/run_eval.py --provider nvidia --model meta/llama-3.3-70b-instruct
```

This runs all 15 cases through the live model and writes a Markdown + JSON report to `eval/results/`. The report includes per-tier accuracy, per-miss breakdown with the model's stated reason, average latency, and an estimated cost per 1,000 runs based on the model's published per-token pricing (free-tier providers show $0).

The eval is small on purpose — 15 hand-written cases, ~$0.01 per full run on Anthropic (free on NVIDIA), ~1 minute wall time. Big enough to surface failure modes, small enough to iterate on between prompt tweaks.

## Deployment

This repo is set up to deploy to **Streamlit Community Cloud** (free tier):

1. Push to GitHub.
2. Go to <https://share.streamlit.io>, point it at this repo, set `app.py` as the entrypoint.
3. Under **App settings → Secrets**, add the backend you're using — either `ANTHROPIC_API_KEY` (paid) or `LLM_PROVIDER=nvidia` + `NVIDIA_API_KEY` + `NVIDIA_MODEL=meta/llama-3.3-70b-instruct` (free tier).

A live URL will be added here once the app is deployed.

## What's intentionally not in here

This is a prototype, not a product. Out of scope on purpose:

- Real sanctions-list lookups (OFAC, EU consolidated list) — licensing and freshness concerns, not a weekend problem.
- Multi-tenant red-flag taxonomies — would need a config layer and tests for each tenant.
- Auth and audit-log persistence — every prediction would be stored with input, output, model version, prompt version, and timestamp in a real deployment.
- Model A/B comparison — a PM prototype should pick one model intentionally; the eval harness is the right place to evaluate model swaps later.

See [`PRODUCT.md`](PRODUCT.md) for the longer "what I'd ship to production differently" treatment.

## License

MIT.
