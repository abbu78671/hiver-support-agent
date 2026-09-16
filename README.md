
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
