import sys
import os
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.pipeline.evaluation_pipeline import EvaluationPipeline

st.set_page_config(page_title="Evaluate Model Performance")

st.title("📊 Evaluate Model Performance")

st.write("""
This page evaluates the NLU system using a labeled evaluation dataset.
Metrics include:
- Accuracy
- Precision
- Recall
- F1-score
""")

if st.button("Run Full Evaluation"):
    with st.spinner("Running evaluation..."):
        try:
            evaluator = EvaluationPipeline()
            metrics = evaluator.run()

            st.success("Evaluation Completed")

            st.subheader("📈 Evaluation Metrics")
            st.json(metrics)

            st.subheader("🧩 Confusion Matrix")
            st.image("artifacts/confusion_matrix.png")

        except Exception as e:
            st.error("Evaluation failed")
            st.exception(e)
