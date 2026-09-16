# SpotifyCares AI Support Agent

### Hiver SDE Intern — Take-Home Assignment

An end-to-end customer-support AI pipeline designed for SpotifyCares, combining intent classification, BM25 retrieval, local LLM response generation, deterministic safety controls, and risk-aware escalation.

---

## Executive Summary

Customer-support automation is not simply a text-generation problem.

A production-oriented support agent must determine:

1. What is the customer asking about?
2. Which historical support cases are relevant?
3. What response can be generated without inventing unsupported actions?
4. Should the request be automated or escalated to a human?

This project implements those stages as an explicit, testable pipeline.

```text
Customer Message
       │
       ▼
┌──────────────────────────┐
│ Intent Classification    │
│ TF-IDF + Logistic Reg.   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ BM25 Retrieval           │
│ Historical Cases         │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Prompt Construction      │
│ Intent + Examples +      │
│ Safety Constraints       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Local LLM Generation     │
│ phi3:mini + Ollama       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Output Safety Guard      │
│ Unsafe-claim detection   │
│ Metadata removal         │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Escalation Policy        │
│ Intent + Confidence +    │
│ Retrieval + Failures     │
└───────────┬──────────────┘
            │
      ┌─────┴─────┐
      ▼           ▼
 AUTO-HANDLE    ESCALATE
 Draft Reply    Human Review