# SpotifyCares AI Support Agent

### Hiver SDE Intern — Take-Home Assignment

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