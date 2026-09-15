"""
Phase 10 - BM25 Retrieval System
Builds retrieval index from non-deflection training conversations.
Evaluates retrieval quality on dev set.
All numbers are OBSERVED.
"""

import json
import pickle
import re
import os
import numpy as np
from rank_bm25 import BM25Okapi

RETRIEVAL_CORPUS_PATH = r"data\processed\retrieval_corpus.jsonl"
DEV_PATH              = r"data\processed\split_dev.jsonl"
INDEX_PATH            = r"data\processed\bm25_index.pkl"
CORPUS_PATH           = r"data\processed\bm25_corpus.pkl"

TOP_K = 3  # number of examples to retrieve per query


def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def tokenize(text):
    """Simple tokenizer: lowercase, remove punctuation, split."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 2]
    return tokens


# ── LOAD RETRIEVAL CORPUS ────────────────────────────────────────
section("1. LOADING RETRIEVAL CORPUS")
corpus = []
with open(RETRIEVAL_CORPUS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        corpus.append(json.loads(line.strip()))
print(f"Retrieval corpus size: {len(corpus):,}")
print(f"(Non-deflection train conversations only)")

# ── BUILD BM25 INDEX ─────────────────────────────────────────────
section("2. BUILDING BM25 INDEX")
print("Tokenizing corpus...")
tokenized_corpus = [tokenize(c['customer_message']) for c in corpus]
avg_tokens = np.mean([len(t) for t in tokenized_corpus])
print(f"Average tokens per message: {avg_tokens:.1f}")

print("Building BM25 index...")
bm25 = BM25Okapi(tokenized_corpus)
print("Index built.")

# Save index and corpus
os.makedirs('data/processed', exist_ok=True)
with open(INDEX_PATH, 'wb') as f:
    pickle.dump(bm25, f)
with open(CORPUS_PATH, 'wb') as f:
    pickle.dump(corpus, f)
print(f"BM25 index saved: {INDEX_PATH}")
print(f"Corpus saved    : {CORPUS_PATH}")


# ── EVALUATE RETRIEVAL QUALITY ───────────────────────────────────
section("3. RETRIEVAL QUALITY — MANUAL INSPECTION")
print(f"Showing top-{TOP_K} retrieved results for 8 dev queries")
print("Manual inspection: are retrieved results relevant?\n")

dev = []
with open(DEV_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        dev.append(json.loads(line.strip()))

# Sample 8 non-deflection dev conversations for inspection
import random
random.seed(42)
non_deflection_dev = [c for c in dev if not c['is_dm_deflection']]
sample_queries = random.sample(non_deflection_dev, min(8, len(non_deflection_dev)))

for i, query_conv in enumerate(sample_queries):
    query = query_conv['customer_message']
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    print(f"\n{'─'*60}")
    print(f"QUERY [{i+1}]: {query[:150]}")
    print(f"{'─'*60}")

    for rank, idx in enumerate(top_indices):
        retrieved = corpus[idx]
        score = scores[idx]
        print(f"\n  Rank {rank+1} (BM25 score: {score:.2f}):")
        print(f"  CUSTOMER : {retrieved['customer_message'][:120]}")
        print(f"  SPOTIFY  : {retrieved['spotify_reply_clean'][:120]}")


# ── SCORE DISTRIBUTION ───────────────────────────────────────────
section("4. RETRIEVAL SCORE DISTRIBUTION")
print("Computing scores for 500 random dev queries...")
sample_dev_500 = random.sample(dev, min(500, len(dev)))
all_top_scores = []

for conv in sample_dev_500:
    tokens = tokenize(conv['customer_message'])
    if not tokens:
        continue
    scores = bm25.get_scores(tokens)
    top_score = scores.max()
    all_top_scores.append(top_score)

scores_arr = np.array(all_top_scores)
print(f"Top-1 BM25 score distribution (500 dev queries):")
print(f"  Mean   : {scores_arr.mean():.2f}")
print(f"  Median : {np.median(scores_arr):.2f}")
print(f"  Min    : {scores_arr.min():.2f}")
print(f"  Max    : {scores_arr.max():.2f}")
print(f"  % > 5  : {(scores_arr > 5).mean():.1%}")
print(f"  % > 10 : {(scores_arr > 10).mean():.1%}")
print(f"  % > 2  : {(scores_arr > 2).mean():.1%}")
print(f"  % = 0  : {(scores_arr == 0).mean():.1%}")

print("\nNote: Score > 5 suggests strong lexical overlap (likely relevant)")
print("      Score < 2 suggests weak match (escalation signal)")


# ── RETRIEVAL FUNCTION FOR PIPELINE ─────────────────────────────
section("5. RETRIEVAL FUNCTION DEMO")
print("This is the function the generation pipeline will call:\n")

def retrieve(query_text, bm25_index, corpus_data, top_k=3):
    """
    Given a customer message, retrieve top_k most similar
    historical SpotifyCares conversations.
    Returns list of dicts with customer_message, spotify_reply, score.
    """
    tokens = tokenize(query_text)
    if not tokens:
        return []
    scores = bm25_index.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            'customer_message': corpus_data[idx]['customer_message'],
            'spotify_reply': corpus_data[idx]['spotify_reply_clean'],
            'bm25_score': float(scores[idx])
        })
    return results

# Demo
demo_query = "My Spotify premium subscription was charged twice this month"
results = retrieve(demo_query, bm25, corpus, top_k=3)
print(f"Query: '{demo_query}'\n")
for i, r in enumerate(results):
    print(f"  Result {i+1} (score: {r['bm25_score']:.2f}):")
    print(f"  Customer: {r['customer_message'][:100]}")
    print(f"  Spotify : {r['spotify_reply'][:100]}")
    print()

section("DONE — paste full output")