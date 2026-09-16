# SpotifyCares AI Support Agent

### Hiver SDE Intern â€” Take-Home Assignment

An end-to-end AI customer-support pipeline built around intent classification, BM25 retrieval, locally hosted LLM generation, deterministic safety validation, and risk-aware escalation.

The project is designed as an engineering prototype for automating repetitive SpotifyCares support interactions while maintaining explicit controls around unsupported claims, account actions, and human escalation.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
  - [Intent Classification](#1-intent-classification)
  - [BM25 Retrieval](#2-bm25-retrieval)
  - [Prompt Construction](#3-prompt-construction)
  - [LLM Generation](#4-llm-generation)
  - [Safety Validation](#5-safety-validation)
  - [Escalation Policy](#6-escalation-policy)
- [Data Pipeline](#data-pipeline)
- [Intent Taxonomy](#intent-taxonomy)
- [Evaluation Methodology](#evaluation-methodology)
- [Golden-Set Results](#golden-set-results)
- [Baseline Comparison](#baseline-comparison)
- [Failure Analysis](#failure-analysis)
- [Generation Quality](#generation-quality)
- [Performance and Latency](#performance-and-latency)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Evaluation Artifacts](#evaluation-artifacts)
- [Engineering Decisions](#engineering-decisions)
- [Testing](#testing)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Project Status](#project-status)

---

# Overview

Customer-support automation is not simply a text-generation problem.

A useful support agent must answer four separate questions:

1. **What is the customer asking about?**
2. **Which historical support examples are relevant?**
3. **What response can be generated without inventing unsupported information?**
4. **Should the request be handled automatically or escalated to a human?**

This project implements those responsibilities as separate stages rather than delegating the complete workflow to an unconstrained LLM.

The current pipeline is:

```text
Customer Message
       |
       v
+-----------------------------+
| Intent Classification       |
| TF-IDF + Logistic Regression|
+-------------+---------------+
              |
              v
+-----------------------------+
| BM25 Retrieval              |
| Historical Support Cases    |
+-------------+---------------+
              |
              v
+-----------------------------+
| Prompt Construction         |
| Intent + Retrieved Evidence |
| + Safety Constraints        |
+-------------+---------------+
              |
              v
+-----------------------------+
| Local LLM Generation        |
| phi3:mini via Ollama        |
+-------------+---------------+
              |
              v
+-----------------------------+
| Deterministic Safety Guard  |
| Unsafe-claim Detection      |
| Metadata Removal             |
+-------------+---------------+
              |
              v
+-----------------------------+
| Escalation Policy           |
| Confidence + Retrieval      |
| + Generation Failure        |
+-------------+---------------+
              |
        +-----+-----+
        |           |
        v           v
   AUTO-HANDLE   ESCALATE
   Draft Reply   Human Review

Problem Statement

The source dataset contains real customer-support conversations from Twitter.

Social-media support introduces several challenges:

messages are short and informal
spelling and grammar vary significantly
multiple issues can appear in a single message
customers often omit important context
historical replies can contain operational or account-specific information
a generative model may produce plausible but unsupported actions

The system therefore combines conventional machine learning, retrieval, local LLM inference, deterministic validation, and policy logic.

The goal is not unrestricted automation.

The goal is to produce a grounded support draft when confidence is sufficient and escalate uncertain or risky cases.

System Architecture
End-to-End Flow

              INPUT
                |
                v
      Customer Support Tweet
                |
                v
    +------------------------+
    | Text Classification    |
    | TF-IDF + Logistic Reg. |
    +-----------+------------+
                |
                | intent + confidence
                v
    +------------------------+
    | BM25 Retrieval         |
    | Historical Support     |
    +-----------+------------+
                |
                | top-k examples
                v
    +------------------------+
    | Prompt Builder          |
    | Customer + Intent +     |
    | Retrieved Evidence      |
    +-----------+-------------+
                |
                v
    +------------------------+
    | phi3:mini / Ollama      |
    | Local Generation        |
    +-----------+-------------+
                |
                | generated reply
                v
    +------------------------+
    | Safety Validation       |
    |                         |
    | â€¢ refund claims         |
    | â€¢ account actions       |
    | â€¢ timelines             |
    | â€¢ leaked metadata       |
    +-----------+-------------+
                |
                v
    +------------------------+
    | Escalation Policy       |
    +-----------+-------------+
                |
          +-----+-----+
          |           |
          v           v
       HANDLE      ESCALATE

Core Components
1. Intent Classification

The classification layer predicts one of six support intents.
MODEL
Representation:
    Word-level TF-IDF
    +
    Character-level TF-IDF

Classifier:
    Logistic Regression

Training:
    Class-balanced

Output:
    Predicted intent
    Prediction probability
    Maximum confidence

Current Configuration
Word n-grams         : 1â€“2
Character n-grams    : 3â€“5
Maximum features     : 35,000
Minimum document df  : 2
Sublinear TF         : Enabled
Regularization C     : 2.0
Class weighting      : balanced

2. BM25 Retrieval

The retrieval layer searches historical SpotifyCares conversations for examples relevant to the incoming customer message.

Current Configuration
Retrieval algorithm : BM25
Corpus size         : 18,076 documents
Top-K               : 3

For every query, the retriever returns:

Customer message
Historical SpotifyCares reply
BM25 relevance score

The retrieved examples are used as contextual evidence for generation.

3. Prompt Construction

The prompt combines:

Customer message
+
Predicted intent
+
Top retrieved support examples
+
Explicit response constraints

The generation instructions emphasize:

concise Twitter-style replies
customer-facing language
no invented account access
no fabricated refunds
no unsupported timelines
no invented policy details
no agent metadata
no extra analysis
no prompt continuation

4. LLM Generation

The generation layer uses:

Model   : phi3:mini
Runtime : Ollama
Mode    : Local inference

Current runtime configuration includes:

Maximum generated tokens : 80
Context window           : 2048
Temperature              : 0.2
Top-p                    : 0.9
Persistent model loading : Enabled

5. Safety Validation

LLM output is not treated as automatically safe.

A deterministic post-generation layer runs after inference.

The safety layer checks for patterns involving:

Refund claims
Account changes
Subscription cancellation
Unsupported processing times
Leaked agent metadata
Historical reply signatures

6. Escalation Policy

The policy layer combines multiple signals instead of using intent alone.

Predicted Intent
      +
Intent Confidence
      +
Retrieval Strength
      +
Generation Failure
      |
      v
Escalation Decision

Data Pipeline

The project uses the:

Customer Support on Twitter

dataset from:

thoughtvector/customer-support-on-twitter

The processing workflow is:

Raw Twitter Conversations
          |
          v
SpotifyCares Filtering
          |
          v
Text Cleaning / Normalization
          |
          v
Train / Dev / Test Splits
          |
          +----------------------+
          |                      |
          v                      v
 Keyword-Labeled Dev       Retrieval Corpus
          |                      |
          v                      v
 TF-IDF Classifier           BM25 Index
          |
          v
 Golden-Set Evaluation

Intent Taxonomy

The current system recognizes six categories.

Intent	Scope
SUB_BILLING	Premium, payment, subscription, billing and trial issues
ACCOUNT_ACCESS	Login, password and account-access issues
APP_TECHNICAL	Application crashes, errors, installation and technical issues
PLAYBACK_ISSUE	Playback, buffering, queue, audio and streaming problems
CONTENT_LIBRARY	Albums, playlists, catalog, missing content and library issues
GENERAL_ENQUIRY	General questions, feedback and ambiguous requests


Evaluation Methodology

The project intentionally separates development benchmarking from final golden-set evaluation.

Development Benchmark

The development split uses deterministic keyword-generated labels.

Golden Evaluation Set

The project contains:

200 examples

with labels for:

True intent
Escalation decision
Escalation reason

Golden-Set Results

Latest observed evaluation:

Metric			Result
Intent accuracy		66.5%
Macro F1		66.2%
Escalation accuracy	76.5%
Escalation precision	69.1%
Escalation recall	71.8%
Escalation F1		70.4%
False auto-handle	11.0%
Automation coverage	61.0%
Generation failures	4 / 200

Intent-Level Metrics

Intent		Precision	Recall		F1
SUB_BILLING	73.3%		94.3%		82.5%
ACCOUNT_ACCESS	88.0%		75.9%		81.5%
APP_TECHNICAL	71.4%		80.6%		75.8%
PLAYBACK_ISSUE	61.5%		75.0%		67.6%
CONTENT_LIBRARY	70.4%		50.0%		58.5%
GENERAL_ENQUIRY	34.5%		28.6%		31.2%
Macro F1					66.2%

Baseline Comparison
System	                       Evaluation Basis		     Macro F1		FalseAuto-Handle
Trivial baseline	       Development benchmark		~17%			34.7%
TF-IDF + Logistic Regression	Development benchmark	       96.7%			17.7%
Current pipeline	        200-example golden set		66.2%			11.0%

Failure Analysis

The golden-set evaluation exposed several important failure modes.

1. GENERAL_ENQUIRY vs Specific Intents

Short messages can contain words strongly associated with another category without actually describing that operational problem.

Example:

"Haha, just a very annoying song in this week's Discover Weekly..."

The classifier associated this with PLAYBACK_ISSUE, while the reviewed label was GENERAL_ENQUIRY.

This demonstrates that lexical overlap does not always represent user intent.

2. High-Confidence Misclassification

Some incorrect predictions receive high classifier confidence.

Example pattern:

Predicted intent : GENERAL_ENQUIRY
Confidence       : 0.95

A high probability therefore does not guarantee semantic correctness.

This is one reason confidence is treated as one signal in the escalation layer rather than as truth.

3. Retrieval Can Propagate Historical Behavior

Historical replies may contain:

outdated wording
operational language
metadata
account-specific claims
agent signatures

Retrieving these examples improves contextual grounding but can also expose the generator to undesirable patterns.

The sanitizer and deterministic output guard were introduced to reduce this risk.

4. LLM Overreach

Local generation can produce unsupported actions.

For example, a model may generate a statement implying that a refund has already been processed even though no payment system is connected.

This is why generation is followed by explicit validation.

5. Escalation Trade-Off

The current system reduces false auto-handling relative to the trivial baseline, but some legitimate auto-handle cases are still escalated.

This creates the expected trade-off:

More automation
        â†•
More conservative escalation

The current implementation prioritizes safer handling over maximum automation coverage.

Generation Quality

A previous LLM-judge evaluation on 30 examples produced:

Overall mean : 2.83 / 5
Median       : 3.0 / 5
Sample size  : 30

The evaluation indicates that the system can produce usable responses but still has substantial room for improvement in:

grounding
relevance
consistency
concise response quality
avoidance of unsupported claims

The judge model used was the same local phi3:mini model through Ollama.

Performance and Latency
Original Golden Evaluation

The original 200-example end-to-end run required approximately:

736.4 minutes

on a CPU-only machine.

The bottleneck was local LLM generation.

Generation Optimization

The generation configuration was subsequently optimized using:

Reduced output token budget
Reduced context window
Persistent Ollama model loading
Explicit stop sequences
Deterministic output validation

Three isolated smoke-test cases after optimization produced approximately:

20â€“28 seconds per generated example

These were local smoke tests rather than a complete benchmark.

Therefore, an updated 200-example end-to-end latency result is intentionally not claimed until that evaluation is rerun.

Repository Structure
hiver-support-agent/
â”‚
â”œâ”€â”€ configs/
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â””â”€â”€ processed/
â”‚
â”œâ”€â”€ evaluation/
â”‚   â”œâ”€â”€ baselines/
â”‚   â”œâ”€â”€ golden_set/
â”‚   â””â”€â”€ reports/
â”‚
â”œâ”€â”€ experiments/
â”‚
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ 07_simple_baseline.py
â”‚   â”œâ”€â”€ 10_sample_golden_set.py
â”‚   â””â”€â”€ 11_evaluate_golden_set.py
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ generation/
â”‚   â”‚   â”œâ”€â”€ generator.py
â”‚   â”‚   â””â”€â”€ __init__.py
â”‚   â”‚
â”‚   â”œâ”€â”€ policy/
â”‚   â”‚   â”œâ”€â”€ escalation.py
â”‚   â”‚   â””â”€â”€ __init__.py
â”‚   â”‚
â”‚   â””â”€â”€ evaluation/
â”‚
â”œâ”€â”€ tests/
â”‚
â”œâ”€â”€ decision_log.md
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ .env.example
â””â”€â”€ README.md
Installation
Requirements

Recommended environment:

Python 3.11+
Ollama
phi3:mini

Install project dependencies:

pip install -r requirements.txt
Quickstart
1. Clone the repository
git clone https://github.com/abbu78671/hiver-support-agent.git
cd hiver-support-agent
2. Install Python dependencies
pip install -r requirements.txt
3. Install Ollama

Install Ollama separately and ensure the local Ollama service is running.

Pull the required model:

ollama pull phi3:mini
4. Prepare the dataset

Download:

thoughtvector/customer-support-on-twitter

and place the source file at:

data/raw/twcs.csv
5. Run the generation smoke test
python src/generation/generator.py

This loads the BM25 retrieval artifacts and exercises local response generation.

6. Run the golden evaluation
python scripts/11_evaluate_golden_set.py

Note: the complete golden-set pipeline is computationally expensive on a CPU-only system because each example performs local LLM inference.

Evaluation Artifacts

Generated evaluation outputs are stored under:

evaluation/reports/

Important files include:

golden_set_results.jsonl
golden_set_summary.json

The JSONL file contains example-level results, including:

row_id
customer_message
true_intent
pred_intent
intent_correct
intent_confidence
true_escalation
pred_escalation
escalation_correct
escalation_reason
retrieval_score
generated_reply
generation_failed
latency_seconds

This makes the evaluation inspectable beyond aggregate metrics.

Engineering Decisions

The major design decisions and their rationale are documented separately in:

decision_log.md

The decision log covers topics including:

intent taxonomy
labeling methodology
baseline construction
classifier selection
retrieval approach
model selection
prompt constraints
output safety
escalation design
evaluation methodology
reproducibility considerations

The purpose is to preserve the reasoning behind technical decisions rather than documenting only the final implementation.

Testing

The repository contains a tests/ directory for regression and component checks.

The main engineering validation loop is:

Unit / Regression Tests
        +
Generation Smoke Tests
        +
Development Benchmark
        +
Golden-Set Evaluation

This separation helps identify whether a change affects:

implementation correctness
classifier behavior
retrieval
generation
escalation
Limitations

This project should be considered an AI engineering prototype, not a production customer-support system.

Current limitations include:

Classification

The current classifier struggles with ambiguous messages, especially around:

GENERAL_ENQUIRY
CONTENT_LIBRARY
PLAYBACK_ISSUE
Labeling

The development benchmark relies on deterministic keyword-generated labels.

These labels are useful for engineering comparison but introduce label noise.

Golden-Set Size

A 200-example golden set provides meaningful qualitative and quantitative feedback but is not large enough to establish production-level reliability.

Generation

A local phi3:mini model can still produce unsupported statements in edge cases.

The deterministic safety layer mitigates known patterns but is not a complete semantic safety solution.

Latency

CPU-only local inference is significantly slower than production GPU-backed or hosted inference.

Retrieval Evaluation

The current project records retrieval scores but does not yet provide a comprehensive independently labeled Recall@K / MRR evaluation.

Future Improvements

Potential next iterations include:

Classification
semantic sentence representations
calibrated confidence
class-specific thresholds
improved ambiguity handling
larger reviewed training data
Retrieval
Recall@K measurement
MRR measurement
semantic reranking
hybrid lexical + semantic retrieval
retrieval confidence calibration
Generation
structured output
stricter grounding
stronger response-quality evaluation
better fallback handling
model benchmarking
Safety
broader unsupported-claim detection
semantic validation
sensitive-intent routing
automated regression tests for unsafe generations
Evaluation
larger independently reviewed golden sets
per-intent error analysis
retrieval-specific evaluation
generation-quality rubric expansion
latency distributions and P95 measurements
Productionization
API service layer
observability
request tracing
structured logging
model/version tracking
monitoring dashboards
human-in-the-loop review tooling
Project Status

Status: Functional AI engineering prototype

The current implementation demonstrates a complete customer-support workflow:

Classification
      â†“
Retrieval
      â†“
Grounded Generation
      â†“
Safety Validation
      â†“
Escalation

The repository emphasizes:

Reproducibility
+
Measurable evaluation
+
Explicit safety controls
+
Transparent failure analysis
+
Documented engineering decisions

It intentionally reports both successful results and known limitations rather than presenting development benchmarks as production-level performance.

Author

Syed Abdul Aziz

B.E. Computer Science Engineering
