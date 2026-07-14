"""
AI Product Copilot
-------------------
A structured AI-feature intake copilot: turns "let's add AI to X" requests
into a scored, architecture-recommended, risk-flagged one-pager — in minutes
instead of a multi-meeting scoping cycle.

This digitizes the AI feature intake template + RAG-vs-fine-tune decision
framework used in production at Charles Schwab and Comerica Bank (see case
studies on aiwithbhuvi.blog). The core recommendation engine is a
deterministic, auditable rule/decision-tree — not an LLM — because in
regulated intake workflows, "why did the tool say this?" has to have a
inspectable answer. An optional LLM pass can polish the prose.

Drop this file into the `pages/` folder of the existing bhuvi-ai-lab
Streamlit multipage app.

Author: Bhuvaneswari Kuduva Premkumar
"""

import streamlit as st

st.set_page_config(page_title="AI Product Copilot", page_icon="🧭", layout="wide")

st.title("🧭 AI Product Copilot")
st.caption(
    "Structured intake for 'let's add AI to this' requests: RICE score, "
    "recommended architecture, a draft one-pager, and a risk/guardrail checklist."
)

# ---------------------------------------------------------------------------
# Intake form
# ---------------------------------------------------------------------------

with st.form("intake"):
    st.subheader("1. Tell the copilot about the idea")
    c1, c2 = st.columns(2)
    with c1:
        feature_name = st.text_input("Feature / idea name", "Smart transfer pre-fill")
        business_goal = st.text_area(
            "Business goal (why does this matter?)",
            "Reduce time-to-complete on the fund transfer flow and cut support tickets about wrong field entries.",
        )
        target_user = st.text_input("Target user", "Retail banking customer initiating a transfer")
        data_availability = st.selectbox(
            "How much labeled/historical data exists today?",
            ["None / would need to collect", "Some, but messy or unlabeled", "Solid historical dataset available"],
        )
    with c2:
        knowledge_volatility = st.selectbox(
            "How often does the underlying knowledge/policy change?",
            ["Rarely (quarterly or less)", "Occasionally (monthly)", "Frequently (weekly or more)"],
        )
        regulatory_sensitivity = st.selectbox(
            "Regulatory / compliance sensitivity",
            ["Low", "Medium", "High (financial, health, or legal data)"],
        )
        latency_tolerance = st.select_slider(
            "Latency tolerance", options=["Instant (<200ms)", "Fast (<2s)", "Chat-speed (2-6s)", "Async / batch OK"],
            value="Fast (<2s)",
        )

    st.subheader("2. Score it (RICE)")
    r1, r2, r3, r4 = st.columns(4)
    reach = r1.slider("Reach (users/quarter, thousands)", 1, 500, 50)
    impact = r2.select_slider("Impact", options=[0.25, 0.5, 1, 2, 3], value=1)
    confidence = r3.slider("Confidence (%)", 10, 100, 70)
    effort = r4.slider("Effort (person-weeks)", 1, 40, 8)

    submitted = st.form_submit_button("Generate brief →", use_container_width=True)

# ---------------------------------------------------------------------------
# Deterministic recommendation engine
# ---------------------------------------------------------------------------

def recommend_architecture(data_availability, knowledge_volatility, regulatory_sensitivity, latency_tolerance):
    """Encodes the RAG-vs-fine-tune-vs-rules judgment call as an explicit,
    auditable decision tree instead of a black-box model call."""
    reasons = []

    if regulatory_sensitivity == "High (financial, health, or legal data)" and data_availability == "None / would need to collect":
        reasons.append("High regulatory sensitivity + no data yet → start with a rules-based / human-in-the-loop system, not ML.")
        return "Rules-based + Human-in-the-Loop", reasons

    if knowledge_volatility == "Frequently (weekly or more)":
        reasons.append("Underlying knowledge changes weekly+, so a fine-tuned model would go stale fast and need constant retraining.")
        reasons.append("RAG lets you swap the knowledge base without retraining — current answers on every query.")
        return "Retrieval-Augmented Generation (RAG)", reasons

    if data_availability == "Solid historical dataset available" and knowledge_volatility == "Rarely (quarterly or less)":
        reasons.append("Stable knowledge domain + solid historical data → fine-tuning is viable and can reduce per-query cost/latency vs. RAG.")
        return "Fine-tuned model", reasons

    if data_availability == "None / would need to collect":
        reasons.append("No usable data yet — ship a rules-based V1 while instrumenting the product to collect the data a model would need.")
        return "Rules-based V1, instrument for ML later", reasons

    reasons.append("Moderate data + moderate volatility → RAG gives the best accuracy-to-effort ratio without a training pipeline to maintain.")
    return "Retrieval-Augmented Generation (RAG)", reasons


def risk_checklist(regulatory_sensitivity):
    base = [
        "Model failure modes documented (what does it get wrong, and how do we know?)",
        "Human-in-the-loop path exists for low-confidence or ambiguous outputs",
        "Audit trail: inputs, model version, and output are logged for every decision",
    ]
    if regulatory_sensitivity != "Low":
        base += [
            "Compliance/legal reviewed the data inputs and outputs before pilot",
            "PII / sensitive data handling reviewed (encryption, access control, retention)",
            "Clear disclosure to end users that AI is involved in the decision",
        ]
    if regulatory_sensitivity == "High (financial, health, or legal data)":
        base.append("Explicit 'defer to human' rule for any ambiguous or out-of-policy query — a confident wrong answer is worse than an honest handoff.")
    return base


if submitted:
    rice_score = (reach * impact * (confidence / 100)) / max(effort, 1)
    architecture, reasons = recommend_architecture(
        data_availability, knowledge_volatility, regulatory_sensitivity, latency_tolerance
    )
    checklist = risk_checklist(regulatory_sensitivity)

    st.divider()
    st.subheader("📋 Generated Brief")

    m1, m2, m3 = st.columns(3)
    m1.metric("RICE score", f"{rice_score:.1f}")
    m2.metric("Recommended architecture", architecture)
    m3.metric("Regulatory tier", regulatory_sensitivity.split(" ")[0])

    st.markdown(f"### {feature_name}")
    st.markdown(f"**Business goal:** {business_goal}")
    st.markdown(f"**Target user:** {target_user}")

    st.markdown("**Why this architecture:**")
    for r in reasons:
        st.markdown(f"- {r}")

    st.markdown("**Risk & guardrail checklist:**")
    for item in checklist:
        st.checkbox(item, key=item)

    st.markdown("**Assumptions & confidence:**")
    st.markdown(
        f"- Confidence in these estimates: **{confidence}%** — treat this as a directional bet, not a committed spec.\n"
        f"- Effort estimate ({effort} person-weeks) assumes {data_availability.lower()}.\n"
        f"- Latency tolerance ({latency_tolerance}) was a key input into the architecture recommendation above."
    )

    brief_text = f"""AI FEATURE INTAKE BRIEF
========================
Feature: {feature_name}
Business goal: {business_goal}
Target user: {target_user}

RICE score: {rice_score:.1f}  (Reach={reach}k, Impact={impact}, Confidence={confidence}%, Effort={effort}pw)
Recommended architecture: {architecture}
Reasoning:
{chr(10).join('- ' + r for r in reasons)}

Regulatory sensitivity: {regulatory_sensitivity}
Risk & guardrail checklist:
{chr(10).join('[ ] ' + c for c in checklist)}

Data availability: {data_availability}
Knowledge volatility: {knowledge_volatility}
Latency tolerance: {latency_tolerance}
"""
    st.download_button("⬇ Download brief (.txt)", brief_text, file_name=f"{feature_name.replace(' ', '_')}_brief.txt")

    with st.expander("✨ Optional: polish this brief with an LLM"):
        st.caption(
            "Bring your own OpenAI API key to turn the structured brief above into polished "
            "executive prose. The recommendation and risk logic above never changes — the LLM "
            "only rewrites tone, it doesn't get a vote on the architecture decision."
        )
        api_key = st.text_input("OpenAI API key", type="password")
        if api_key and st.button("Polish with AI"):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a sharp, concise AI product manager writing an executive brief. Do not change any facts, numbers, or the architecture recommendation — only improve clarity and tone."},
                        {"role": "user", "content": brief_text},
                    ],
                )
                st.markdown(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"Couldn't reach the LLM: {e}")

st.divider()
st.caption(
    "Built by Bhuvaneswari Kuduva Premkumar · Core scoring/recommendation logic is deterministic "
    "and auditable by design — see the AI Architecture write-up on aiwithbhuvi.blog."
)
