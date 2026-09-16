# SpotifyCares AI Support Agent

> Hiver SDE Intern Take-Home Assignment — AI Customer Support Agent

An end-to-end AI support pipeline for SpotifyCares that classifies 
customer tweets, retrieves grounded historical resolutions, 
generates draft replies, and decides whether to auto-handle or 
escalate to a human agent — with evidence-backed evaluation.

---

## Table of Contents

- [What This Builds](#what-this-builds)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Project Structure](#project-structure)
- [Intent Taxonomy](#intent-taxonomy)
- [Escalation Policy](#escalation-policy)
- [Evaluation Methodology](#evaluation-methodology)
- [Baselines](#baselines)
- [Failure Analysis](#failure-analysis)
- [What Is Misleading About the Headline Number](#what-is-misleading-about-the-headline-number)
- [Limitations](#limitations)
- [What I Would Do With One More Week](#what-i-would-do-with-one-more-week)
- [Dataset](#dataset)
- [Decision Log](#decision-log)
- [Sources](#sources)

---

## What This Builds

An AI customer support agent for **SpotifyCares** that:

1. **Classifies** incoming customer tweets into 6 intents derived from real data
2. **Retrieves** the top-3 most similar historical SpotifyCares resolutions (BM25, 18,685 cases)
3. **Decides** whether to auto-handle or escalate — with a stated reason
4. **Generates** grounded draft replies using phi3:mini anchored to retrieved examples

---

## Results

### Intent Classification — 200-example golden set

| Intent | Precision | Recall | F1 |
|---|---|---|---|
| SUB_BILLING | 76.7% | 94.3% | 84.6% |
| ACCOUNT_ACCESS | 92.3% | 77.4% | 84.2% |
| APP_TECHNICAL | 68.8% | 71.0% | 69.8% |
| PLAYBACK_ISSUE | 63.9% | 71.9% | 67.6% |
| CONTENT_LIBRARY | 70.0% | 55.3% | 61.8% |
| GENERAL_ENQUIRY | 33.3% | 33.3% | 33.3% |
| **Macro F1** | | | **66.9%** |

### Escalation Policy

| Metric | Value |
|---|---|
| False auto-handle rate | **9.0%** (trivial baseline: 34.7%) |
| Escalation recall | 76.9% |
| Escalation F1 | 69.4% |
| Automation coverage | 61.0% |

### Response Quality

| Metric | Value |
|---|---|
| Mean judge score | 2.83 / 5.0 |
| Sample size | 30 auto-handled replies |
| Judge model | phi3:mini via Ollama |

### Baselines

| System | Macro F1 | False Auto-Handle |
|---|---|---|
| Trivial — majority class + always auto-handle | ~17% | 34.7% |
| Simple — TF-IDF + LR (dev set, keyword labels) | 90%* | 17.8%* |
| **Final system — golden set** | **66.9%** | **9.0%** |

> *Dev set metrics use keyword pseudo-labels — not directly comparable to golden set

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Intent Classifier | TF-IDF + Logistic Regression (scikit-learn) |
| Retrieval | BM25 Okapi (rank-bm25) |
| Generation | phi3:mini via Ollama (local, zero cost) |
| LLM Judge | phi3:mini via Ollama |
| Language Detection | langdetect |
| Data Processing | pandas, numpy |
| Testing | pytest (16 tests) |
| Dataset | Kaggle: thoughtvector/customer-support-on-twitter |

Customer tweet
│
▼
┌─────────────────────────┐
│ TF-IDF + Logistic │ → predicted intent
│ Regression Classifier │ → confidence score
└─────────────────────────┘
│
▼
┌─────────────────────────┐
│ Escalation Policy │ SUB_BILLING / ACCOUNT_ACCESS → always ESCALATE
│ (Evidence-backed) │ confidence < 0.50 → ESCALATE
│ │ retrieval score < 5.0 → ESCALATE
└─────────────────────────┘
│
├──── ESCALATE → { decision, reason }
│ No generation. Human agent handles.
│
└──── AUTO_HANDLE
│
▼
┌─────────────────────┐
│ BM25 Retrieval │ → top-3 similar historical
│ (18,685 docs) │ SpotifyCares conversations
└─────────────────────┘
│
▼
┌─────────────────────┐
│ phi3:mini │ → grounded draft reply
│ Generation │ anchored to retrieved examples
└─────────────────────┘
│
▼
Draft reply


**Design principle**: Escalation runs before generation.
Cases that should escalate never reach the LLM — saving compute
and preventing bot responses on sensitive issues.

---

## Quickstart

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed locally
- phi3:mini model pulled

```bash
# Pull the model (one time, ~2GB)
ollama pull phi3:mini

# Start Ollama (keep this running in a separate terminal)
ollama serve
```

### Option A — Fast path (under 15 minutes)

Uses pre-built artifacts committed to the repository.

```bash
git clone https://github.com/abbu78671/hiver-support-agent.git
cd hiver-support-agent
pip install -r requirements.txt
python scripts/11_evaluate_golden_set.py
```

Results appear in `evaluation/reports/golden_set_summary.json`

### Option B — Full rebuild (4+ hours)

Download the dataset first:
- Kaggle: `thoughtvector/customer-support-on-twitter`
- Place `twcs.csv` in `data/raw/twcs.csv`

Then run scripts in order:

```bash
python scripts/01_inspect_data.py
python scripts/04_build_conversations.py
python scripts/05_split_dataset.py
python scripts/06_trivial_baseline.py
python scripts/07_simple_baseline.py
python scripts/08_build_retrieval.py
python scripts/09_fix_preprocessing.py
python scripts/10_sample_golden_set.py
python scripts/11_evaluate_golden_set.py
python scripts/12_llm_judge.py
```

### Demo — Test on a single message

```bash
python scripts/demo.py
```

This runs 5 test messages through the full pipeline interactively.

### Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

hiver-support-agent/
├── README.md
├── requirements.txt
├── decision_log.md ← 10 engineering decisions with evidence
├── .env.example
├── configs/
│ ├── brand.yaml ← brand and data config
│ ├── model.yaml ← classifier, retrieval, generation config
│ └── eval.yaml ← evaluation config
├── data/
│ ├── processed/
│ │ ├── bm25_index.pkl ← pre-built BM25 index (committed)
│ │ ├── bm25_corpus.pkl ← retrieval corpus (committed)
│ │ └── split_test_flagged.jsonl
│ └── samples/
│ └── spotify_sample_100.jsonl
├── scripts/
│ ├── 01_inspect_data.py ← dataset inspection
│ ├── 04_build_conversations.py← thread reconstruction
│ ├── 05_split_dataset.py ← train/dev/test split
│ ├── 06_trivial_baseline.py ← majority class baseline
│ ├── 07_simple_baseline.py ← TF-IDF + LR baseline
│ ├── 08_build_retrieval.py ← BM25 index construction
│ ├── 09_fix_preprocessing.py ← language filter + near-dup detection
│ ├── 10_sample_golden_set.py ← golden set sampling
│ ├── 11_evaluate_golden_set.py← full pipeline evaluation
│ ├── 12_llm_judge.py ← response quality evaluation
│ └── demo.py ← interactive demo
├── src/
│ ├── generation/
│ │ └── generator.py ← BM25 retrieval + phi3:mini generation
│ └── policy/
│ └── escalation.py ← evidence-backed escalation policy
├── tests/
│ ├── test_escalation.py ← 7 escalation policy tests
│ ├── test_retrieval.py ← 9 retrieval and tokenization tests
│ └── test_classifier.py ← 4 classifier interface tests
├── evaluation/
│ ├── golden_set/
│ │ └── golden_set_labelled.csv ← 200-example evaluation set
│ ├── baselines/
│ │ ├── tfidf_vectorizer.pkl
│ │ └── lr_classifier.pkl
│ └── reports/
│ ├── golden_set_summary.json ← headline metrics
│ ├── golden_set_results.jsonl ← per-example results
│ └── judge_results.json ← response quality scores
└── experiments/
└── experiment_log.jsonl ← 5 experiments logged



---

## Intent Taxonomy

Derived from keyword frequency analysis of 40,728 SpotifyCares
conversations. Not assumed — observed from real data.

| Intent | Definition | Example |
|---|---|---|
| PLAYBACK_ISSUE | Cannot play, stream, or download music | "Songs keep stopping randomly" |
| APP_TECHNICAL | App crashes, bugs, device compatibility | "App won't open on iPhone after update" |
| SUB_BILLING | Premium, billing, charges, cancellation | "Charged twice this month" |
| ACCOUNT_ACCESS | Cannot log in, password, account recovery | "Forgot my password, can't get in" |
| CONTENT_LIBRARY | Missing songs, removed albums, metadata | "Album disappeared from my library" |
| GENERAL_ENQUIRY | Feature requests, feedback, praise, unclear | "When will you add this feature?" |

### Ambiguous Boundaries (documented)

- **PLAYBACK vs APP_TECHNICAL**: App crash → APP_TECHNICAL. Music won't play but app works → PLAYBACK_ISSUE
- **ACCOUNT vs SUB_BILLING**: Can't log in → ACCOUNT_ACCESS. Logged in but wrong plan → SUB_BILLING
- **CONTENT vs PLAYBACK**: Song exists but won't play → PLAYBACK. Song missing entirely → CONTENT_LIBRARY

---

## Escalation Policy

Evidence-backed — derived from observed escalation rates on dev set.

```python
def escalation_decision(intent, confidence, retrieval_score, failed):

    # Rule 1: High-risk intents → always ESCALATE
    # Evidence: 70.9% and 70.1% real escalation rate observed
    if intent in ['SUB_BILLING', 'ACCOUNT_ACCESS']:
        return 'ESCALATE', f'High-risk intent: {intent}'

    # Rule 2: Low classifier confidence → ESCALATE
    if confidence < 0.50:
        return 'ESCALATE', f'Low confidence: {confidence:.2f}'

    # Rule 3: No good historical precedent → ESCALATE
    if retrieval_score < 5.0:
        return 'ESCALATE', 'Insufficient historical evidence'

    # Rule 4: Generation failed → ESCALATE
    if failed:
        return 'ESCALATE', 'Reply generation failed'

    return 'AUTO_HANDLE', f'Intent={intent}, conf={confidence:.2f}'
```

| Intent | Observed Escalation Rate | Policy |
|---|---|---|
| SUB_BILLING | 70.9% | Always escalate |
| ACCOUNT_ACCESS | 70.1% | Always escalate |
| APP_TECHNICAL | 20.5% | Threshold-based |
| PLAYBACK_ISSUE | 19.5% | Threshold-based |
| CONTENT_LIBRARY | 14.0% | Threshold-based |
| GENERAL_ENQUIRY | 27.3% | Threshold-based |

---

## Evaluation Methodology

### Golden Set Construction

- 200 examples sampled from test split (never seen during training)
- Stratified sampling across all 6 intents
- 14 deliberately ambiguous cases included
- Near-duplicate detection: BM25 score > 50 vs training corpus → excluded
- Language filter: non-English messages excluded

### Data Splits

| Split | Size | Purpose |
|---|---|---|
| Train | 28,509 (70%) | Classifier training + retrieval corpus |
| Dev | 6,109 (15%) | Threshold tuning |
| Test | 6,110 (15%) | Golden set source |
| Golden | 200 | Final evaluation only |

**Split unit**: conversation-level (not tweet-level).
Zero conversation ID overlap confirmed across all splits.

### Metrics

- Intent: accuracy, macro F1, per-class precision/recall/F1
- Escalation: precision, recall, F1, false auto-handle rate, coverage
- Response: LLM judge score 1–5 on 30 auto-handled replies

---

## Failure Analysis

### 1. GENERAL_ENQUIRY is nearly unclassifiable
- **Example**: "Haha, just a very annoying song this week" → predicted PLAYBACK_ISSUE
- **Cause**: Catch-all category shares vocabulary with every other intent
- **F1**: 33.3%

### 2. Mid-conversation tweets lack context
- **Example**: "MacOS 10.12.6 - view scale is set to actual size" → predicted GENERAL_ENQUIRY
- **Cause**: Follow-up messages are meaningless without conversation history (49.2% of conversations are multi-turn)

### 3. ACCOUNT_ACCESS escalation rule is too broad
- **Example**: "Did regression testing miss the login redirect loop?" → escalated unnecessarily
- **Cause**: 70% blanket rule escalates technical observations alongside real emergencies

### 4. CONTENT_LIBRARY vs PLAYBACK_ISSUE boundary is ambiguous
- **CONTENT_LIBRARY recall**: 55.3% — 17 cases misclassified
- **Cause**: "Song won't play" and "Song is missing" share vocabulary

### 5. Persistent issues not detected within low-risk intents
- **Example**: "Spotify keeps deleting my downloaded songs" → AUTO_HANDLE (PLAYBACK, conf 0.86)
- **Cause**: Policy has no signal for repeated or persistent issues

---

## What Is Misleading About the Headline Number

**Macro F1 of 66.9% has four specific problems:**

1. **GENERAL_ENQUIRY at 33.3% F1 drags macro down.**
   The five operationally important intents average 73.6% F1.
   Macro weights all classes equally — a misleading choice when
   one class is a poorly-defined catch-all.

2. **Baseline comparison uses different label systems.**
   Simple baseline (90%*) was measured on dev set with keyword
   pseudo-labels. Final system (66.9%) was measured on golden set
   with independent labels. Not directly comparable.

3. **9.0% false auto-handle = 18 real customers.**
   Those 18 customers received bot responses when they needed
   a human. The business cost of those 18 errors is unknown.

4. **Response quality of 2.83/5.0 is between poor and acceptable.**
   The generation component is the weakest part of the system.

---

## Limitations

- phi3:mini (3.8B) produces mediocre reply quality (mean 2.83/5.0)
- CPU inference averages 68.9s per example — not production-viable
- BM25 retrieval cannot handle semantic similarity (paraphrases score low)
- System classifies single tweets — no multi-turn conversation context
- No grounding verification — cannot prove replies use retrieved content

---

## What I Would Do With One More Week

1. **Multi-turn context** — prepend prior conversation turns to classifier input
2. **Embedding retrieval comparison** — measure whether sentence-transformers improve CONTENT_LIBRARY recall specifically
3. **Urgency detection** — add signal for "this has happened multiple times" to escalation policy
4. **Grounding verification** — check whether generated reply references retrieved content or invents new information
5. **Threshold optimization** — grid search confidence and retrieval thresholds on dev set

---

## Dataset

- **Source**: Kaggle `thoughtvector/customer-support-on-twitter`
- **Brand selected**: SpotifyCares
- **Selection evidence**: Highest clean volume (43,265 brand tweets), 0% duplicate rate, 100% English, tight domain (5–6 intents)
- **Total conversations built**: 40,728 customer→brand pairs
- **Retrieval corpus**: 18,685 non-deflection train conversations

---

## Decision Log

See [`decision_log.md`](decision_log.md) for 10 engineering decisions,
each with alternative considered and evidence for the choice made.

Key decisions:
- Why SpotifyCares over AppleSupport, Delta, Uber
- Why BM25 over sentence-transformer embeddings
- Why TF-IDF + LR over fine-tuned model
- Why conversation-level splitting over tweet-level
- Why escalation runs before generation

---

## Sources

| Source | Used For |
|---|---|
| Kaggle: thoughtvector/customer-support-on-twitter | Primary dataset |
| Robertson et al. (1994) — BM25 | Retrieval method |
| scikit-learn (Pedregosa et al., 2011) | TF-IDF + LR classifier |
| rank-bm25 library | BM25 implementation |
| langdetect library | Language filtering |
| phi3:mini — Microsoft (2024) via Ollama | Generation and judging |
| Hiver SDE Intern Assignment brief | Problem framing |



---

## Architecture
