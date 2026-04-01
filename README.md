# Sentiment Analysis — Binary Classification

## Overview
This project evaluates and compares multiple NLP models for binary sentiment classification
(Positive vs. Negative) on a Twitter dataset. It covers conventional ML pipelines and 
pre-trained transformer models, benchmarking them on accuracy, precision, recall, F1-score, 
and inference speed.

---

## Dataset
- **Source:** [Kaggle — Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)
- **File used:** `train.csv`
- **Labels:** Negative (0) vs. Positive (1/4) — Neutral samples excluded
- **Sample size:** 15,000 tweets (random_state=42)

---

## Models Evaluated

### Conventional ML
| Vectorizer        | Classifier          |
|-------------------|---------------------|
| Bag of Words      | Logistic Regression |
| TF-IDF            | Logistic Regression |
| N-Grams (1,2)     | Logistic Regression |
| Bag of Words      | Linear SVM          |
| TF-IDF            | Linear SVM          |
| N-Grams (1,2)     | Linear SVM          |

### Deep Learning (Zero-Shot Inference)
| Model       | HuggingFace ID                                        |
|-------------|-------------------------------------------------------|
| DistilBERT  | `distilbert-base-uncased-finetuned-sst-2-english`     |
| RoBERTa     | `cardiffnlp/twitter-roberta-base-sentiment`           |

---

## Results Summary

| Model                  | Accuracy | F1-Score | Time (s) |
|------------------------|----------|----------|----------|
| RoBERTa (Transformer)  | 0.9067   | 0.9231   | 2.4859   |
| LogReg + TF-IDF        | 0.8733   | 0.8795   | 0.1811   |
| SVM + TF-IDF           | 0.8690   | 0.8752   | 0.1811   |
| DistilBERT (Transformer)| 0.8400  | 0.8621   | 1.5728   |

> Full results in `Sentiment_Analysis_Report.docx`

---

## Setup & Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch transformers accelerate
```

---

## Usage
```bash
# Place train.csv in the working directory, then run:
python sentiment_analysis.py
```

---

## Key Findings
- **RoBERTa** achieved the best accuracy (90.67%) due to Twitter-specific fine-tuning
- **LogReg + TF-IDF** is the best conventional model — fast and accurate (87.33%)
- **DistilBERT** underperformed due to domain mismatch (trained on movie reviews)
- Conventional ML models are up to **13x faster** than transformers at inference

---

## Project Structure
```
├── train.csv                        # Dataset (download from Kaggle)
├── NLP_Sentiment_Analysis.ipynb           # Main experiment script
├── Sentiment_Analysis_Report.docx   # Full report with analysis
└── README.md
```

---

## References
- Shriv, A. (2021). Sentiment Analysis Dataset. Kaggle.  
  https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset
- Cardiff NLP. `twitter-roberta-base-sentiment`. HuggingFace.
- Hugging Face. `distilbert-base-uncased-finetuned-sst-2-english`.
- scikit-learn documentation: https://scikit-learn.org


##Authors
- Kolimi Heena Farheen
- Samridhi Rauthan
