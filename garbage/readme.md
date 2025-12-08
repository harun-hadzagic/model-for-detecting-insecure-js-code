Used the pre-trained CodeBERT tokenizer (microsoft/codebert-base) to convert raw code into numerical tokens (input_ids and attention_mask).

Why Subword Tokenization (CodeBERT)?
For code, standard text tokenization is often insufficient because it breaks code structure (e.g., separating db.query into two tokens).

We chose to use the CodeBERT tokenizer for the following reasons:

Semantic Understanding: CodeBERT was pre-trained on billions of lines of code and natural language, allowing it to capture the unique syntax and semantics of programming languages.

Handling Unseen Data: The subword (BPE) approach allows the model to tokenize unseen variable names or new functions by breaking them into known sub-units, which is essential for code vulnerability detection.

Transformer Compatibility: The output format (input_ids and attention_mask) is the standard required for fine-tuning the powerful CodeBERT Transformer architecture using PyTorch.

# 🛡️ CodeGuard: JavaScript Vulnerability Detector (CodeBERT Fine-Tuning)

**CodeGuard** is a machine learning–powered tool for detecting security vulnerabilities (such as **SQL Injection**, **XSS**, and **Command Injection**) in JavaScript/Node.js code.  
It leverages **Transfer Learning** by fine-tuning **[CodeBERT](https://huggingface.co/microsoft/codebert-base)** for a **binary classification** task — classifying code as:

> ✅ SECURE  
> ❌ INSECURE (Vulnerable)

The resulting model achieved **exceptionally high performance** on unseen test data, making it a strong candidate for integration into static analysis pipelines.

---

## 📊 Performance Results

After fine-tuning the CodeBERT model for **5 epochs**, the model achieved the following results on the held-out test set:

| **Metric** | **Result** | **Goal** | **Interpretation** |
|-------------|------------|----------|--------------------|
| **Test Accuracy** | 0.9827 | High | High overall correctness |
| **Test Precision (Insecure)** | 0.9814 | High | Low rate of False Alarms (False Positives) |
| **Test Recall (Insecure)** | 0.9840 | Highest Priority | Excellent. Only 1.6% of vulnerabilities missed |
| **Test F1-Score** | 0.9827 | High | Strong balance between Precision and Recall |

---

## 🧠 Methodology: How We Built the Detector

### 1. Data Generation and Curation (5,091 Pairs)
- **What:** Created pairs of JavaScript code snippets — each pair containing an **insecure implementation** (label `1`) and a **secure/fixed version** (label `0`).  
- **Why:** This paired dataset teaches the model both *what a vulnerability looks like* and *how it should be fixed*.  
- **Focus Areas:**  
  - SQL Injection  
  - Command Injection  
  - Path Traversal  
  - Weak Cryptography  
  - Cross-Site Scripting (XSS)

---

### 2. Tokenization and Data Splitting
- **What:** Converted raw code into numeric vectors using the **CodeBERT tokenizer**, which understands programming syntax.  
- **Why:** Neural networks require numerical input.  
- **Data Split:** Training / Validation / Test sets, ensuring secure–insecure pairs remain in the same set for fairness.  
- **Generated Files:**  

train_data.npz
val_data.npz
test_data.npz


---

### 3. Model Training (Fine-Tuning)
- **Model:** `microsoft/codebert-base`
- **Frameworks:** PyTorch + Hugging Face Transformers  
- **Training Goal:** Optimize for **recall_insecure** — prioritizing detection of *all* vulnerabilities (minimizing false negatives).  
- **Output Directory:**

results_codebert_js/
└── final_best_model/
*(Contains saved fine-tuned model weights.)*

---

## ⚙️ Getting Started

### 🧩 Prerequisites
You’ll need **Python 3.8+** and the following dependencies:

```bash
pip install torch transformers datasets scikit-learn
pip install "accelerate>=0.26.0"  # Required by Hugging Face Trainer
