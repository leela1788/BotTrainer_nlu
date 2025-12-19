🤖 BotTrainer – LLM-Based NLU Model Trainer & Evaluator

BotTrainer is an LLM-powered Natural Language Understanding (NLU) system designed for chatbot applications.
Instead of training traditional machine learning models, this project uses prompt engineering and few-shot learning with modern Large Language Models (LLMs) to perform intent classification and evaluate performance using standard ML metrics.

🚀 Key Features

🔍 Intent Classification using LLMs (LLaMA 3 via Groq – free & fast)

🧠 Prompt Engineering & Few-Shot Learning

📊 Model Evaluation with Accuracy, Precision, Recall, F1-score

📉 Confusion Matrix Visualization

🧪 Structured JSON Output Enforcement

🖥️ Interactive Multi-Page Streamlit UI

🧩 Modular, ML-style Project Architecture

📝 Centralized Logging & Custom Exception Handling

🛠️ Tech Stack

Python 3.10+

Groq API (LLaMA 3.1 models)

Streamlit (UI)

Scikit-learn (evaluation metrics)

PyYAML (config management)

Matplotlib (visualization)

📂 Project Structure
BotTrainer_nlu/
├── app/
│   ├── streamlit_app.py
│   └── pages/
│       ├── 1_Test_Intent_Classification.py
│       └── 2_Evaluate_Model_Performance.py
│
├── data/
│   ├── intents.json
│   └── eval_data.json
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── prompt_builder.py
│   │   ├── llm_inference.py
│   │   └── intent_classifier.py
│   │
│   ├── pipeline/
│   │   ├── inference_pipeline.py
│   │   └── evaluation_pipeline.py
│   │
│   ├── config_manager.py
│   ├── logging_system.py
│   └── exceptions.py
│
├── config/
│   └── config.yaml
│
├── artifacts/
│   └── logs/
│
├── requirements.txt
├── setup.py
└── README.md

🧠 How This Project Works
🔹 Important Concept

The LLM is NOT trained in this project.

Instead:

A pre-trained LLM is used

The task is defined using prompt engineering

The system is evaluated like a traditional ML model

🧩 Components Explained
1️⃣ intents.json

Defines the intent space

Provides few-shot examples

Used to ground the LLM

2️⃣ Prompt Engineering

Intent definitions + examples are injected into prompts

Enforces strict JSON output

Replaces feature engineering and model training

3️⃣ Inference Pipeline

Builds prompt

Calls LLM

Parses structured response

Returns predicted intent + confidence

4️⃣ Evaluation Pipeline

Uses eval_data.json as ground truth

Compares predicted intent vs actual intent

Computes:

Accuracy

Precision

Recall

F1-score

Generates confusion matrix

📊 Evaluation Strategy

Closed-set intent classification

Same intent labels as intents.json

Different (unseen) user utterances in eval_data.json

Metrics are meaningful and ML-correct

Intent labels in eval_data.json are never shown to the LLM — they are used only for scoring.

🖥️ Streamlit Application

The UI is implemented as a multi-page Streamlit app:

🧠 Page 1 – Test Intent Classification

Enter any sentence

View predicted intent and confidence

Real-time LLM inference

📊 Page 2 – Evaluate Model Performance

Run full evaluation on labeled dataset

View metrics

Visualize confusion matrix

Run the app:

streamlit run app/streamlit_app.py

⚙️ Configuration
📄 config/config.yaml
llm:
  provider: groq
  model: llama-3.1-8b-instant
  temperature: 0.2
  max_tokens: 300

paths:
  intents_path: data/intents.json
  eval_path: data/eval_data.json

evaluation:
  confidence_threshold: 0.6

🔐 Environment Setup

1️⃣ Create virtual environment:

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


2️⃣ Install dependencies:

pip install -r requirements.txt


3️⃣ Create .env file:

GROQ_API_KEY=your_api_key_here

📝 Logging & Error Handling

Centralized logging stored in artifacts/logs/

Timestamped log files for each run

Custom NLUException for clean error propagation

🎯 Learning Outcomes

Through this project, you learn:

LLM-based NLU system design

Prompt engineering as a replacement for ML training

Intent classification pipelines

ML-style evaluation of LLM systems

Streamlit app development

Logging, configuration, and error handling

Clean, scalable project structuring

📌 Future Enhancements

Entity extraction & slot filling

Entity-level evaluation metrics

Unknown intent (OOD) detection

Error analysis dashboard

Docker deployment
