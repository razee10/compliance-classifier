"""Streamlit UI for the compliance classifier.

Run locally:
    streamlit run app.py

Provider/model can be chosen at runtime from the sidebar. The dropdown only
shows providers whose API key is available (in `.env`, the shell env, or
Streamlit Cloud secrets). Env defaults (`LLM_PROVIDER`, `ANTHROPIC_MODEL`,
`NVIDIA_MODEL`) still drive the initial selection.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from classifier import (
    ANTHROPIC_MODEL_DEFAULT,
    DEFAULT_PROVIDER,
    NVIDIA_MODEL_DEFAULT,
    _build_client,
    classify,
)

load_dotenv()

st.set_page_config(
    page_title="Compliance Classifier",
    page_icon=":mag:",
    layout="centered",
)

st.title("Compliance Document Classifier")
st.caption(
    "First-pass triage for KYC / AML / sanctions-adjacent documents. "
    "Returns a category, confidence, and red flags with evidence quoted from the source."
)

# --- Provider catalog. Model lists are convenience presets; users can override with "Custom...".
PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "key_env": "ANTHROPIC_API_KEY",
        "models": [
            "claude-sonnet-4-6",
            "claude-opus-4-7",
            "claude-haiku-4-5-20251001",
        ],
        "default_model": ANTHROPIC_MODEL_DEFAULT,
    },
    "nvidia": {
        "key_env": "NVIDIA_API_KEY",
        "models": [
            "meta/llama-3.3-70b-instruct",
        ],
        "default_model": NVIDIA_MODEL_DEFAULT,
    },
}

# Pull API keys from Streamlit Cloud secrets into env if not already set, so the
# SDKs pick them up transparently.
for _cfg in PROVIDERS.values():
    _key_env = _cfg["key_env"]
    if not os.environ.get(_key_env):
        try:
            os.environ[_key_env] = st.secrets[_key_env]
        except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
            pass

available_providers = [
    p for p, cfg in PROVIDERS.items() if os.environ.get(cfg["key_env"])
]

if not available_providers:
    st.error(
        "No API key found for any supported provider. "
        "Set `ANTHROPIC_API_KEY` or `NVIDIA_API_KEY` in `.env` locally, "
        "or in Streamlit Cloud secrets."
    )
    st.stop()

EXAMPLES = {
    "Clear negative": (
        "ACME Industries Ltd. Incorporated in Delaware, US. Three directors, all US residents. "
        "Operating since 2012. Annual revenue ~$12M. Standard commercial cleaning services."
    ),
    "AML / complex structure": (
        "Company Alpha Trading SA. Registered in Panama. Five directors, three listed as "
        "residents of jurisdictions flagged by FATF. UBO structure unclear -- two intermediate "
        "holding entities in the BVI."
    ),
    "Possible structuring": (
        "Transaction of EUR 9,800 from Client X's account to a new beneficiary not previously seen. "
        "No stated purpose."
    ),
    "Adversarial (news mention)": (
        "GreenLeaf Logistics Ltd, UK-incorporated freight forwarder operating in Manchester. "
        "Recent news article noted that one of their competitors has been criticised for serving "
        "clients in Iran. GreenLeaf has no overseas operations."
    ),
    "Complex case (long-form memo)": (
        """1. Corporate Identity and Registration

Alpha Trading SA (hereinafter "the Subject") is a legal entity organized and existing under the laws of the Republic of Panama, registered under Folio No. 1558292. The company's stated primary business objective is "international logistics and commodities brokerage," specifically focusing on energy sector derivatives within emerging markets. While the Subject maintains a registered office in Panama City, initial digital footprint analysis suggests a lack of significant physical operational infrastructure commensurate with its declared multi-million dollar turnover.

2. Governance and Management Composition

The Subject is governed by a Board of Directors consisting of five (5) appointed individuals. A cross-reference of residency permits and tax IDs reveals a high concentration of geographic risk. While the Chairman (Marcus Thorne) and the Secretary (Ana Sophia Rivera) are residents of Panama and Spain respectively, the remaining three directors maintain primary residences in jurisdictions currently subject to increased monitoring by the FATF (Financial Action Task Force).

- Director A (Yaroslav Petrov): Resident of a high-risk jurisdiction in Central Eurasia.
- Director B (Li Na): Resident of a jurisdiction flagged for strategic AML/CFT deficiencies.
- Director C (Omar Al-Fayed): Resident of a region currently on the FATF Grey List.

The presence of a majority (60%) of the board in high-risk zones poses a significant challenge for ongoing monitoring and the verification of "fit and proper" status.

3. Ownership Complexity and Layering

The Ultimate Beneficial Ownership (UBO) structure of Alpha Trading SA is currently classified as Obscured. The Subject's share capital is not held directly by natural persons but is partitioned through a series of offshore vehicles:

- Primary Shareholder: 100% of Alpha Trading SA is owned by Apex Prime Holdings Ltd, registered in the British Virgin Islands (BVI).
- Secondary Layer: Apex Prime Holdings Ltd is, in turn, a subsidiary of Global Shell Foundations Inc, also domiciled in the BVI.

Initial inquiries to the BVI Registry have failed to produce a Register of Members or a clear Declaration of Trust. The use of two consecutive intermediate holding entities in a "tax neutral" jurisdiction is a classic indicator of Layering, intended to decouple the assets from the beneficial owner. Consequently, the identity of the natural persons who ultimately own or control 25% or more of the Subject cannot be verified with the documentation currently on file."""
    ),
}

with st.sidebar:
    st.subheader("Examples")
    pick = st.selectbox("Load an example", options=["-- choose --"] + list(EXAMPLES.keys()))

    st.subheader("Backend")
    default_provider_idx = (
        available_providers.index(DEFAULT_PROVIDER)
        if DEFAULT_PROVIDER in available_providers
        else 0
    )
    selected_provider = st.selectbox(
        "Provider",
        options=available_providers,
        index=default_provider_idx,
        help="Only providers with an API key in env/secrets appear here.",
    )

    provider_cfg = PROVIDERS[selected_provider]
    preset_models = provider_cfg["models"]
    default_model = provider_cfg["default_model"]
    model_options = preset_models + ["Custom..."]
    default_model_idx = (
        preset_models.index(default_model) if default_model in preset_models else 0
    )
    model_choice = st.selectbox(
        "Model",
        options=model_options,
        index=default_model_idx,
        help="Pick a preset or 'Custom...' to paste any model ID the provider accepts.",
        key=f"model_choice_{selected_provider}",
    )
    if model_choice == "Custom...":
        selected_model = st.text_input(
            "Custom model ID",
            value=default_model,
            help=(
                "For NVIDIA: any NIM-hosted model ID (e.g. `meta/llama-3.1-405b-instruct`). "
                "For Anthropic: any model your key has access to."
            ),
            key=f"custom_model_{selected_provider}",
        ).strip()
    else:
        selected_model = model_choice

if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if pick != "-- choose --" and pick in EXAMPLES:
    st.session_state.doc_text = EXAMPLES[pick]

doc_text = st.text_area(
    "Document text",
    value=st.session_state.doc_text,
    height=220,
    placeholder="Paste a company profile, transaction summary, or KYC extract...",
)

run = st.button(
    "Classify",
    type="primary",
    disabled=not (doc_text.strip() and selected_model),
)

if run:
    client = _build_client(selected_provider)
    with st.spinner(f"Running two-prompt pipeline on `{selected_provider}` / `{selected_model}`..."):
        result = classify(
            doc_text,
            client=client,
            provider=selected_provider,
            model=selected_model,
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Label", result.label)
    col2.metric("Confidence", result.confidence)
    col3.metric("Latency", f"{result.latency_ms} ms")

    if result.confidence == "low":
        st.warning(
            "Low confidence: in production this case would route to a human reviewer rather than auto-action."
        )

    st.markdown(f"**Primary reason:** {result.primary_reason or '_(none provided)_'}")

    st.subheader("Red flags")
    if not result.red_flags:
        st.info("No red flags identified.")
    else:
        for flag in result.red_flags:
            severity_icon = {"critical": ":red_circle:", "warning": ":large_orange_diamond:", "info": ":information_source:"}.get(
                flag.severity, ""
            )
            grounded_note = "" if flag.evidence_grounded else "  _(evidence NOT found in input -- model may have hallucinated)_"
            st.markdown(
                f"{severity_icon} **{flag.flag}** ({flag.severity}){grounded_note}\n\n"
                f"> {flag.evidence}"
            )

    if result.unknowns:
        st.subheader("Reviewer should verify")
        for u in result.unknowns:
            st.markdown(f"- {u}")

    if result.errors:
        with st.expander("Pipeline notes"):
            for e in result.errors:
                st.text(e)

    with st.expander("Raw model output (audit trail)"):
        st.json(
            {
                "provider": result.provider,
                "model": result.model,
                "classify": result.raw_classify,
                "flags": result.raw_flags,
                "tokens": {
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                },
            }
        )
