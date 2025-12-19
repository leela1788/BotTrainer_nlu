import sys
import os
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


st.set_page_config(
    page_title="BotTrainer – LLM-Based NLU",
    layout="centered"
)

st.title("🤖 BotTrainer – LLM-Based NLU System")

st.markdown("""
### Welcome!

This application demonstrates a **prompt-engineered NLU system** built using **LLMs (LLaMA 3 via Groq)**.

### Available Pages:
- 🧠 **Test Intent Classification**
- 📊 **Evaluate Model Performance**

Use the sidebar to navigate between pages.
""")

st.info("Select a page from the left sidebar 👈")
