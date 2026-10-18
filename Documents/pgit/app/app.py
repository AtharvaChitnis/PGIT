"""Streamlit dashboard: campaign targeting + customer-feedback auto-fill.

Run from the repo root:
    python -m src.train            # creates models/campaign_model.joblib
    streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import config
from src.evaluation import Economics
from src.feedback import analyze_comment, load_survey_filler
from src.predict import load_bundle, recommend

st.set_page_config(page_title="Campaign Targeting", page_icon="📈", layout="wide")


@st.cache_resource
def get_bundle(path: str):
    return load_bundle(path)


@st.cache_resource
def get_filler(path: str):
    return load_survey_filler(path)


# ---------------------------------------------------------------- sidebar: economics + model
st.sidebar.header("Campaign economics")
st.sidebar.caption("Assumptions you control. Defaults are illustrative, not real campaign data.")
contact_cost = st.sidebar.number_input("Cost per contact", min_value=0.0, value=1.0, step=0.5)
conversion_value = st.sidebar.number_input("Net value per conversion", min_value=0.01, value=10.0, step=1.0)
use_cap = st.sidebar.checkbox("Limit number of contacts (capacity)")
max_contacts = st.sidebar.number_input("Max contacts", min_value=1, value=1000, step=100) if use_cap else None
model_path = st.sidebar.text_input("Model file", str(config.BUNDLE_PATH))

economics = Economics(contact_cost, conversion_value)
st.sidebar.metric("Break-even response probability", f"{economics.break_even_probability:.1%}")

try:
    bundle = get_bundle(model_path)
except FileNotFoundError:
    st.error("No trained model found. Run `python -m src.train` first (see README).")
    st.stop()

if not bundle.get("calibrated", False):
    st.warning(
        "This model was trained without probability calibration. Expected-value decisions "
        "assume calibrated probabilities, so treat recommendations with caution."
    )

st.title("Campaign targeting")
tab_single, tab_batch, tab_feedback, tab_report = st.tabs(
    ["Single customer", "Batch scoring", "Customer feedback", "Model report"]
)

# ---------------------------------------------------------------- single customer
with tab_single:
    st.caption(
        "Enter pre-contact customer details. Call duration is deliberately not a feature "
        "because it is only known after the call."
    )
    values: dict = {}
    with st.form("customer_form"):
        cols = st.columns(3)
        for i, name in enumerate(bundle["numeric"]):
            s = bundle["numeric_summary"][name]
            is_int = float(s["min"]).is_integer() and float(s["max"]).is_integer()
            with cols[i % 3]:
                if is_int:
                    values[name] = st.number_input(
                        name, min_value=int(s["min"]), max_value=int(s["max"]), value=int(s["median"])
                    )
                else:
                    values[name] = st.number_input(
                        name, min_value=s["min"], max_value=s["max"], value=s["median"]
                    )
        for i, name in enumerate(bundle["categorical"]):
            with cols[i % 3]:
                values[name] = st.selectbox(name, bundle["categories"][name])
        comment = st.text_area("Customer comment (optional)", "")
        submitted = st.form_submit_button("Get recommendation")

    if submitted:
        row = pd.DataFrame([values])
        if bundle.get("text_column"):
            row[bundle["text_column"]] = comment
        result = recommend(bundle, row, economics).iloc[0]
        p, ev = float(result["response_probability"]), float(result["expected_value"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Response probability", f"{p:.1%}")
        c2.metric("Expected value of contact", f"{ev:+.2f}")
        c3.metric("Break-even probability", f"{economics.break_even_probability:.1%}")
        if bool(result["contact"]):
            st.success("Recommendation: CONTACT")
        else:
            st.info("Recommendation: DO NOT CONTACT")

        if comment.strip():
            info = analyze_comment(comment, get_filler(str(config.SURVEY_MODEL_PATH)))
            st.subheader("Feedback signal")
            st.write(info["survey_answers"])
            if not bundle.get("text_column"):
                st.caption(
                    "This model was trained without text, so the comment does not change the "
                    "recommendation above. Shown for context only."
                )

# ---------------------------------------------------------------- batch scoring
with tab_batch:
    upload = st.file_uploader("CSV of customers (';' or ',' separated)", type="csv")
    if upload is not None:
        data = pd.read_csv(upload, sep=None, engine="python")
        try:
            scored = recommend(bundle, data, economics, max_contacts)
        except ValueError as exc:
            st.error(str(exc))
        else:
            n = int(scored["contact"].sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Customers", len(scored))
            m2.metric("Recommended contacts", n)
            m3.metric("Expected net value", f"{scored.loc[scored['contact'], 'expected_value'].sum():,.0f}")
            st.dataframe(scored.sort_values("expected_value", ascending=False), use_container_width=True)
            st.download_button("Download scored CSV", scored.to_csv(index=False), "scored_customers.csv")

# ---------------------------------------------------------------- feedback auto-fill
with tab_feedback:
    st.caption(
        "Fills structured survey answers from free text. The bundled model is trained on "
        "VADER-derived labels, so treat its output as illustrative."
    )
    text = st.text_area("Customer comment", "I am so happy with this purchase — amazing quality and fast support!")
    if st.button("Auto-fill survey"):
        if not text.strip():
            st.warning("Enter a comment first.")
        else:
            out = analyze_comment(text, get_filler(str(config.SURVEY_MODEL_PATH)))
            st.write(f"Method: {out.get('method', 'model')}")
            left, right = st.columns(2)
            left.subheader("Sentiment")
            left.json(out["sentiment"])
            right.subheader("Survey answers")
            right.json(out["survey_answers"])

# ---------------------------------------------------------------- model report
with tab_report:
    metrics_file = config.REPORTS_DIR / "metrics.json"
    if metrics_file.exists():
        rep = json.loads(metrics_file.read_text())
        st.write(f"Selected model: **{rep['selected_model']}**")
        st.subheader("Held-out test metrics (evaluated once)")
        st.json(rep["test_metrics"])
        st.subheader("Model comparison (train CV / validation)")
        st.dataframe(pd.DataFrame(rep["model_comparison"]).drop(columns=["best_params"]))
        sweep_file = config.REPORTS_DIR / "threshold_sweep_oof.csv"
        if sweep_file.exists():
            sweep = pd.read_csv(sweep_file).set_index("threshold")
            st.subheader("Threshold sweep (out-of-fold, development data)")
            st.line_chart(sweep[["precision", "recall", "f1"]])
    else:
        st.info("No reports/metrics.json yet. Run `python -m src.train`.")
