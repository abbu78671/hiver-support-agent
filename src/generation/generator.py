"""
src/generation/generator.py
Grounded reply generation using phi3:mini via Ollama + BM25 retrieval.
"""

import json
import pickle
import re
import time
import subprocess
import numpy as np


BM25_INDEX_PATH  = "data/processed/bm25_index.pkl"
CORPUS_PATH      = "data/processed/bm25_corpus.pkl"
OLLAMA_MODEL     = "phi3:mini"
TOP_K_RETRIEVAL  = 3
MAX_REPLY_TOKENS = 200
OLLAMA_TIMEOUT   = 120


def tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [t for t in text.split() if len(t) > 2]


def load_retrieval_components():
    with open(BM25_INDEX_PATH, 'rb') as f:
        bm25 = pickle.load(f)
    with open(CORPUS_PATH, 'rb') as f:
        corpus = pickle.load(f)
    return bm25, corpus


def sanitize_text(text, max_len=200):
    if not isinstance(text, str):
        return ""
    injection_patterns = [
        r'your task[:\s]',
        r'construct a',
        r'you are now',
        r'ignore (previous|above|prior)',
        r'new instruction',
        r'system prompt',
        r'disregard',
        r'forget (everything|all)',
        r'additional constraints',
        r'adhering strictly',
    ]
    text_lower = text.lower()
    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            return "[retrieved example removed: contained unsafe content]"
    return text[:max_len]


def retrieve(query_text, bm25, corpus, top_k=TOP_K_RETRIEVAL):
    tokens = tokenize(query_text)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            'customer_message': sanitize_text(corpus[idx]['customer_message']),
            'spotify_reply':    sanitize_text(corpus[idx]['spotify_reply_clean']),
            'bm25_score':       float(scores[idx])
        })
    return results


def build_prompt(customer_message, intent, retrieved_examples):
    examples_text = ""
    for i, ex in enumerate(retrieved_examples, 1):
        examples_text += (
            f"\nExample {i}:\n"
            f"Customer: {ex['customer_message'][:200]}\n"
            f"SpotifyCares replied: {ex['spotify_reply'][:200]}\n"
        )

    prompt = f"""You are a SpotifyCares support agent replying on Twitter.

STRICT RULES:
1. NEVER promise a refund, credit, or any account action
2. NEVER say you have reviewed or checked the account
3. NEVER invent dollar amounts, timelines, or policy details
4. ONLY use information visible in the examples below
5. If examples show asking for more info, do the same
6. Keep reply to 1-2 short sentences maximum
7. Match SpotifyCares tone: friendly, direct, brief
8. Do NOT start with I
9. Do NOT add any notes or explanations after the reply

CUSTOMER INTENT: {intent}

EXAMPLES:
{examples_text}
CUSTOMER MESSAGE: {customer_message}

SPOTIFYCARES REPLY:"""

    return prompt


def call_ollama(prompt, model=OLLAMA_MODEL, timeout=OLLAMA_TIMEOUT):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": MAX_REPLY_TOKENS,
            "temperature": 0.3,
            "top_p": 0.9,
        }
    })
    start = time.time()
    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST",
             "http://localhost:11434/api/generate",
             "-H", "Content-Type: application/json",
             "-d", payload],
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout
        )
        latency = time.time() - start
        if result.returncode != 0:
            return None, latency
        response = json.loads(result.stdout)
        raw = response.get('response', '').strip()
        for marker in ['(Note:', '(note:', 'Note:', '\n(']:
            if marker in raw:
                raw = raw[:raw.index(marker)].strip()
        return raw, latency
    except Exception:
        return None, time.time() - start


def generate_reply(customer_message, intent, bm25, corpus):
    retrieved = retrieve(customer_message, bm25, corpus)
    top_score = retrieved[0]['bm25_score'] if retrieved else 0.0
    prompt = build_prompt(customer_message, intent, retrieved)
    reply, latency = call_ollama(prompt)

    return {
        'customer_message':    customer_message,
        'intent':              intent,
        'retrieved_examples':  retrieved,
        'top_retrieval_score': top_score,
        'prompt':              prompt,
        'generated_reply':     reply,
        'latency_seconds':     round(latency, 2),
        'model':               OLLAMA_MODEL,
        'generation_failed':   reply is None,
    }


if __name__ == "__main__":
    print("Loading retrieval components...")
    bm25, corpus = load_retrieval_components()

    test_cases = [
        ("My Spotify keeps crashing every time I open it on iPhone",
         "APP_TECHNICAL"),
        ("I was charged twice for premium this month",
         "SUB_BILLING"),
        ("The album I saved yesterday is now gone from my library",
         "CONTENT_LIBRARY"),
    ]

    for msg, intent in test_cases:
        print(f"\n{'='*60}")
        print(f"CUSTOMER: {msg}")
        print(f"INTENT: {intent}")
        result = generate_reply(msg, intent, bm25, corpus)
        print(f"RETRIEVED (top score: {result['top_retrieval_score']:.1f}):")
        for ex in result['retrieved_examples']:
            print(f"  [{ex['bm25_score']:.1f}] {ex['customer_message'][:80]}")
        print(f"REPLY: {result['generated_reply']}")
        print(f"LATENCY: {result['latency_seconds']}s")