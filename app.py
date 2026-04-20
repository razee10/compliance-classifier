"""Streamlit UI for the compliance classifier.

Run locally:
    streamlit run app.py

Set `LLM_PROVIDER=anthropic` (default) or `LLM_PROVIDER=nvidia` in `.env`.
Each provider needs its own API key -- see `.env.example`.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from classifier import DEFAULT_MODEL, DEFAULT_PROVIDER, _build_client, classify

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

# --- API key sanity check. Actual resolution happens inside _build_client / the SDK.
# Streamlit Cloud users can also put keys in `.streamlit/secrets.toml` and os.environ
# will pick them up automatically if mirrored there.
required_key = "NVIDIA_API_KEY" if DEFAULT_PROVIDER == "nvidia" else "ANTHROPIC_API_KEY"
if not os.environ.get(required_key):
    # Try Streamlit secrets as a fallback.
    try:
        os.environ[required_key] = st.secrets[required_key]
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass

if not os.environ.get(required_key):
    st.error(
        f"No `{required_key}` found. Set it in `.env` locally, "
        f"or in Streamlit secrets when deployed. "
        f"Active provider: `{DEFAULT_PROVIDER}`."
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
}

with st.sidebar:
    st.subheader("Examples")
    pick = st.selectbox("Load an example", options=["-- choose --"] + list(EXAMPLES.keys()))
    st.subheader("Backend")
    st.code(f"provider: {DEFAULT_PROVIDER}\nmodel:    {DEFAULT_MODEL}", language=None)

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

run = st.button("Classify", type="primary", disabled=not doc_text.strip())

if run:
    client = _build_client(DEFAULT_PROVIDER)
    with st.spinner("Running two-prompt pipeline..."):
        result = classify(doc_text, client=client)

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
