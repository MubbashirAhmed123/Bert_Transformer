# 🤖 Customer Support Intent Classifier using BERT

A fine-tuned BERT model that classifies customer support messages into 6 intent categories. Built with HuggingFace Transformers and PyTorch.

---

## 📌 Overview

This project fine-tunes `bert-base-uncased` to automatically detect the **intent** behind a customer support message — enabling smarter chatbots and helpdesk automation.

**Example:**
```
Input:  "I want my money back"
Output: refund_request ✅

Input:  "where is my package"
Output: order_status ✅
```

---

## 🎯 Supported Intents

| Label | Intent | Example Phrase |
|-------|--------|----------------|
| 0 | `greeting` | "hello", "hi there", "good morning" |
| 1 | `password_reset` | "forgot my password", "can't log in" |
| 2 | `refund_request` | "I want a refund", "money back please" |
| 3 | `order_status` | "where is my order", "track my package" |
| 4 | `payment_issue` | "card declined", "payment failed" |
| 5 | `complaint` | "very bad service", "I am unhappy" |

---

## 🛠️ Tech Stack

- **Python** 3.8+
- **PyTorch** — model training & inference
- **HuggingFace Transformers** — BERT tokenizer & model
- **Scikit-learn** — train/val split
- **Pandas** — data handling

---

## ⚙️ How It Works

### Pipeline

```
Raw Data (240 phrases)
       ↓
Train/Val Split (80/20)
       ↓
BertTokenizer (padding, truncation, max_len=32)
       ↓
BertForSequenceClassification (6 labels)
       ↓
Fine-tuning (5 epochs, AdamW lr=2e-5)
       ↓
predict(text) → intent label
```

### Training Details

| Parameter | Value |
|-----------|-------|
| Base model | `bert-base-uncased` |
| Number of labels | 6 |
| Max token length | 32 |
| Batch size | 4 |
| Epochs | 5 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Train/Val split | 80% / 20% |
| Device | CUDA (if available) else CPU |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install torch transformers scikit-learn pandas
```

### 2. Run the training script

```bash
python intent_classifier.py
```

### 3. Use the predict function

```python
from intent_classifier import predict

print(predict("hello"))                          # greeting
print(predict("i need to reset my password"))    # password_reset
print(predict("where is my package"))            # order_status
print(predict("payment not working"))            # payment_issue
print(predict("i want my money back"))           # refund_request
print(predict("this is terrible service"))       # complaint
```

---

## 📊 Dataset

- ~40 labeled examples per intent
- 240 total phrases
- Manually curated real-world customer support style sentences

---

## 🧠 Key Concepts

**BERT** (Bidirectional Encoder Representations from Transformers) is a pre-trained NLP model by Google that reads text in both directions simultaneously, giving it a deep understanding of context.

**Fine-tuning** means taking BERT's pre-trained weights and training it further on your specific task (intent classification) with a small dataset.

**Self-Attention** is the mechanism inside BERT that lets every word in a sentence focus on every other word — understanding relationships like "bank" referring to a riverbank vs a financial bank based on context.


