import sys
import os
import streamlit as st

# ✅ Add project root to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.pipeline.inference_pipeline import InferencePipeline



st.set_page_config(page_title="Test Intent Classification")

st.title("🧠 Test Intent Classification")

st.write("Enter a sentence to predict its intent using the LLM-based NLU system.")

@st.cache_resource
def load_pipeline():
    return InferencePipeline()

pipeline = load_pipeline()

user_query = st.text_input(
    "Enter a sentence:",
    placeholder="e.g., Book a cab to the airport"
)

if st.button("Predict Intent"):
    if user_query.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Running inference..."):
            try:
                result = pipeline.run(user_query)

                st.success("Prediction Successful")
                st.subheader("🔍 Result")
                st.json(result)

            except Exception as e:
                st.error("Error during inference")
                st.exception(e)
