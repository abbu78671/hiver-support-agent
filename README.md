# SpotifyCares AI Support Agent
### Hiver SDE Intern Take-Home Assignment

---

## Quickstart — Reproduce Results in Under 15 Minutes

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
# Kaggle: thoughtvector/customer-support-on-twitter
# Place twcs.csv in: data/raw/twcs.csv

# 3. Run evaluation on pre-built artifacts
python scripts/11_evaluate_golden_set.py

# 4. View results
cat evaluation/reports/golden_set_summary.json
```

> **Note**: Full pipeline (data processing → evaluation) takes ~4 hours on CPU.
> The above command uses pre-built artifacts (BM25 index, classifier, golden set)
> and runs in under 15 minutes.

---

## What This Builds

An AI customer support agent for **SpotifyCares** that:

1. **Classifies** incoming customer tweets into 6 intents derived from real data
2. **Retrieves** similar historical SpotifyCares resolutions (BM25, 18,685 cases)
3. **Generates** grounded draft replies using phi3:mini + retrieved examples
4. **Decides** whether to auto-handle or escalate, with a stated reason

---

## Results

### Intent Classification (200-example golden set)

| Intent | Precision | Recall | F1 |
|---|---|---|---|
| SUB_BILLING | 76.7% | 94.3% | 84.6% |
| ACCOUNT_ACCESS | 92.3% | 77.4% | 84.2% |
| APP_TECHNICAL | 68.8% | 71.0% | 69.8% |
| PLAYBACK_ISSUE | 63.9% | 71.9% | 67.6% |
| CONTENT_LIBRARY | 70.0% | 55.3% | 61.8% |
| GENERAL_ENQUIRY | 33.3% | 33.3% | 33.3% |
| **Macro F1** | | | **66.9%** |

### Baselines

| System | Macro F1 | False Auto-Handle |
|---|---|---|
| Trivial (majority class) | ~17% | 34.7% |
| Simple (TF-IDF + LR, dev set) | 90%* | 17.8%* |
| **Final system** | **66.9%** | **9.0%** |

*Measured on dev set with keyword pseudo-labels — not directly comparable

### Escalation Policy

| Metric | Value |
|---|---|
| False auto-handle rate | **9.0%** (vs 34.7% trivial) |
| Escalation recall | 76.9% |
| Automation coverage | 61.0% |

### Response Quality (LLM Judge, n=30)

| Dimension | Score (1-5) |
|---|---|
| Overall mean | 2.83 |
| Median | 3.0 |

> Judge model: phi3:mini via Ollama (local, zero cost).

---

## System Architecture