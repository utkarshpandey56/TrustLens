# TrustLens: An Interpretable Non-Agentic Machine Learning System for Automated Misinformation Detection

**IntraTech 1.0 at IIT Tirupati (2026) — NLP & Trust Safety Track (Team 97)**

**Built by:** [@Anish796](https://github.com/Anish796), [@Pranav041207](https://github.com/Pranav041207) & [@utkarshpandey56](https://github.com/utkarshpandey56)

A non-agentic natural language processing pipeline designed to identify fake news articles while delivering token-level local explanations to human reviewers. By combining selective transformer layer fine-tuning with local interpretable model-agnostic explanations (LIME), the system bridges the gap between complex deep learning models and actionable decision support for fact-checking workflows.

> **Important:** The system provides model predictions, not verified fact-checking. A prediction of "fake" or "real" should not be treated as definitive proof of whether a claim is true or false.

## 1. Overview

The TrustLens system analyzes submitted news content and predicts whether it is:

- **Likely Misinformation**

- **Likely Authentic**

For every prediction, the system provides:

- Prediction label

- Confidence score

- Token-level local explanations (LIME)

- Review status

- Reviewer-oriented Streamlit dashboard

The goal is to support human reviewers such as fact-checkers, researchers, journalists, and content moderators by helping them prioritize content that may require further investigation.

## 2. Key Features

### Content Analysis

Users can submit a news headline and article text through the Streamlit interface.

The system processes the text using the fine-tuned transformer classification model and generates a prediction with a confidence score.

### Explainable Predictions

The system uses `LimeTextExplainer` to generate feature attributions for individual predictions. LIME constructs a sparse linear surrogate model around the prediction vector, assigning relative weights to tokens that push the classification towards "Authentic" or "Misinformation".

### Review Dashboard

The Streamlit dashboard maintains analyzed items and displays them according to their predicted category and confidence.

The dashboard supports:

- Likely misinformation items ranked by confidence

- Likely real items ranked by confidence

- Review status tracking

### Human-in-the-Loop Review

Incoming news items are triaged based on model output probabilities. Items falling within an uncertain probability threshold are flagged for priority human review. Reviewers can confirm, dismiss, or override flags, with feedback logged non-destructively to an external audit database for subsequent active learning cycles.

## 3. System Architecture

```
                    ┌─────────────────────┐
                    │      User Input     │
                    │  Title + Article    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Regex Dateline &    │
                    │ spaCy Preprocessing │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Partial Transformer │
                    │ Fine-Tuning (BERT)  │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │   Prediction &   │      │   Token-Level    │
        │    Confidence    │      │    LIME Layer    │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    ┌─────────────────────┐
                    │   Streamlit Review  │
                    │      Dashboard      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Human Reviewer      │
                    │ Audit Database / HITL│
                    └─────────────────────┘

```

## 4. Project Structure

```
fake-news-detection/
│
├── app.py
│
├── model_loader.py
│
├── explainability.py
│
├── requirements.txt
│
├── README.md
│
├── trained_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── ...
│
└── data/
    └── WELFake.csv

```

### File Descriptions

| File / Folder       | Purpose                                         |
| ------------------- | ----------------------------------------------- |
| `app.py`            | Main Streamlit application and review dashboard |
| `model_loader.py`   | Loads the trained model and tokenizer           |
| `explainability.py` | Generates token-level LIME explanations         |
| `trained_model/`    | Saved trained transformer model and tokenizer   |
| `data/`             | WELFake dataset used for training/evaluation    |
| `requirements.txt`  | Python dependencies                             |
| `README.md`         | Project documentation                           |

## 5. Machine Learning Pipeline & Data Engineering

### Step 1 — Dataset and Artifact Stripping

Model training was executed on the **WELFake** benchmark dataset, comprising 72,134 raw news records. Initial exploratory analysis identified critical dataset artifacts—specifically dateline headers such as "WASHINGTON (Reuters) -" present predominantly in real news articles which allow models to learn shortcut features rather than semantic context. A regular expression pipeline was deployed to strip metadata headers prior to tokenization.

### Step 2 — Preprocessing

To handle large text corpora efficiently under memory constraints, preprocessing was implemented using a chunked iteration pattern via **spaCy** (`batch_size=500`). By explicitly disabling unnecessary syntactic parsing and named entity recognition modules, the system achieved lightweight lemmatization and stop-word filtering across 63,676 deduplicated records without memory degradation.

### Step 3 — Model Architecture & Partial Freezing

The core classifier utilizes `bert-base-uncased` initialized with pre-trained weights. To balance representation capacity with computational efficiency, a partial freezing strategy was applied:

- **Bottom 10 encoder layers (\~90M parameters):** Frozen to retain general language syntax.

- **Upper 2 transformer layers (Layers 10-11) & Classification Head (\~19.5M parameters):** Unfrozen for fine-tuning.

Training was conducted using the **AdamW optimizer** with a learning rate of $2 \times 10^{-5}$, a weight decay of $0.02$, and a classifier dropout rate of $0.3$ to prevent overfitting. Gradient norm clipping was enforced at $1.0$.

### Step 4 — Explainability Layer

TrustLens integrates `LimeTextExplainer` to generate feature attributions for individual predictions, assigning relative weights to tokens pushing classifications toward authentic or misinformation classes.

## 6. Training & Evaluation

The architecture was evaluated on a held-out 20% test split (12,735 articles) against baseline methods, achieving an accuracy of **97.78%** and an F1-score of **0.98**.

| Model Architecture                  | Accuracy   | Precision | Recall   | F1-Score | AUC      |
| ----------------------------------- | ---------- | --------- | -------- | -------- | -------- |
| TF-IDF + Logistic Regression        | 88.4%      | 0.87      | 0.89     | 0.88     | 0.91     |
| Static BERT (Linear Head Only)      | 91.2%      | 0.90      | 0.91     | 0.91     | 0.94     |
| **TrustLens (Top-2 Unfrozen BERT)** | **97.78%** | **0.98**  | **0.97** | **0.98** | **0.99** |

## 7. Installation

### Requirements

- Python 3.10+

- PyTorch

- Transformers

- spaCy

- Streamlit

- LIME

- Scikit-learn

- Pandas / NumPy

### Clone and Setup

```
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd fake-news-detection
python -m venv venv
venv\Scripts\activate  # On Unix: source venv/bin/activate
pip install -r requirements.txt

```

## 8. Running the Application

Ensure the trained model is placed inside `trained_model/`, then launch the dashboard:

```
streamlit run app.py

```

## 9. Responsible Use & Limitations

TrustLens is intended as a **decision-support tool**, not an automated fact-checker. Predictions depend on training distribution, model representations, and regex artifact stripping. Human review remains essential for final determinations.

## 10. Team & Event

Built for **IntraTech 1.0** at **IIT Tirupati** (2026) — **Team 97** (NLP & Trust Safety Track)

- [@Anish796](https://github.com/Anish796)

- [@Pranav041207](https://github.com/Pranav041207)

- [@utkarshpandey56](https://github.com/utkarshpandey56)
