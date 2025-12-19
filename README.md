# BotTrainer – LLM-Based NLU Model Trainer & Evaluator

A professional tool for intent classification in Natural Language Understanding (NLU) systems, leveraging Large Language Models (LLMs) through prompt engineering without traditional model training.

## Project Overview

BotTrainer is an innovative NLU system that utilizes LLaMA 3 via the Groq API for accurate intent classification. Unlike conventional approaches that require extensive model training, this project employs advanced prompt engineering techniques to achieve high-performance intent recognition. The system provides an intuitive Streamlit-based interface for testing classifications and evaluating model performance against labeled datasets.

## Key Features

- **Prompt-Engineered Classification**: Utilizes LLaMA 3 for intent classification without model training
- **Real-Time Inference**: Fast, on-demand intent prediction for user queries
- **Comprehensive Evaluation**: Automated performance metrics including accuracy, precision, recall, and F1-score
- **Visual Analytics**: Confusion matrix visualization for detailed performance analysis
- **Streamlit UI**: User-friendly web interface for testing and evaluation
- **Configurable Setup**: YAML-based configuration for easy customization
- **Robust Logging**: Comprehensive logging system for monitoring and debugging
- **Error Handling**: Custom exception handling for reliable operation

## Tech Stack

- **Language**: Python 3.8+
- **LLM Provider**: Groq API (LLaMA 3.1-8B Instant)
- **UI Framework**: Streamlit
- **Data Processing**: PyYAML, scikit-learn
- **Visualization**: Matplotlib
- **Environment Management**: python-dotenv

## Project Structure

```
.
├── app/
│   ├── streamlit_app.py          # Main Streamlit application
│   └── pages/
│       ├── 1_Test_Intent_Classification.py    # Intent testing interface
│       └── 2_Evaluate_Model_Performance.py    # Performance evaluation interface
├── src/
│   ├── __init__.py
│   ├── config_manager.py         # Configuration loading utilities
│   ├── exceptions.py             # Custom exception classes
│   ├── logging_system.py         # Logging configuration
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py     # Data loading and preprocessing
│   │   ├── intent_classifier.py  # Intent classification logic
│   │   ├── llm_inference.py      # LLM API interaction
│   │   └── prompt_builder.py     # Prompt engineering utilities
│   └── pipeline/
│       ├── __init__.py
│       ├── evaluation_pipeline.py # Model evaluation workflow
│       └── inference_pipeline.py  # Inference workflow
├── config/
│   └── config.yaml               # Application configuration
├── data/
│   ├── intents.json              # Intent definitions and examples
│   └── eval_data.json            # Labeled evaluation dataset
├── artifacts/                    # Generated outputs (metrics, plots)
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # Project documentation
```

## How It Works

1. **Data Ingestion**: Loads intent definitions and evaluation data from JSON files
2. **Prompt Engineering**: Constructs optimized prompts for the LLM based on available intents
3. **LLM Inference**: Sends engineered prompts to LLaMA 3 via Groq API for classification
4. **Intent Classification**: Processes LLM responses to extract predicted intents and confidence scores
5. **Evaluation**: Compares predictions against ground truth using standard ML metrics

The system operates through two main pipelines:
- **Inference Pipeline**: Handles real-time intent classification requests
- **Evaluation Pipeline**: Performs batch evaluation on labeled datasets

## Evaluation Strategy

The evaluation process assesses model performance using:
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of prediction errors across intent classes

Evaluation runs on a labeled dataset (`data/eval_data.json`) and generates:
- Metrics JSON file in `artifacts/`
- Predictions log for detailed analysis
- Confusion matrix plot for visual inspection

## Streamlit UI Pages

### Home Page
- Project introduction and navigation guide
- Overview of available features

### Test Intent Classification
- Interactive text input for user queries
- Real-time intent prediction with confidence scores
- JSON-formatted results display

### Evaluate Model Performance
- One-click evaluation execution
- Comprehensive metrics dashboard
- Confusion matrix visualization
- Downloadable evaluation artifacts

## Setup & Run Instructions

### Prerequisites
- Python 3.8 or higher
- Groq API key

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd llm_nlu_project
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Configure the application**:
   Edit `config/config.yaml` to customize settings (optional).

### Running the Application

Start the Streamlit application:
```bash
streamlit run app/streamlit_app.py
```

Access the application at `http://localhost:8501`

## Logging & Error Handling

### Logging
- Comprehensive logging system implemented in `src/logging_system.py`
- Logs pipeline operations, API calls, and errors
- Configurable log levels for different environments

### Error Handling
- Custom `NLUException` class for NLU-specific errors
- Graceful handling of API failures and data loading issues
- User-friendly error messages in the Streamlit interface
- Detailed error logging for debugging

## Future Enhancements

- **Multi-Model Support**: Integration with additional LLM providers (OpenAI, Anthropic)
- **Fine-Tuning Capabilities**: Option to fine-tune models on custom datasets
- **Advanced Prompt Engineering**: Dynamic prompt optimization based on performance feedback
- **Batch Processing**: Support for bulk intent classification
- **API Endpoints**: RESTful API for integration with external systems
- **Model Comparison**: Side-by-side evaluation of different LLM configurations
- **Data Augmentation**: Techniques to expand training data through synthetic generation
- **Real-time Monitoring**: Dashboard for tracking model performance over time
