"""
Phase 10b - Fix two data quality issues discovered during retrieval:
1. Non-English messages (Indonesian slipped through ASCII filter)
2. Near-duplicate detection and flagging

Run this BEFORE building the golden set.
"""

import json
import re
import os

try:
    from langdetect import detect, LangDetectException
except ImportError:
    print("Installing langdetect...")
    os.system("pip install langdetect")
    from langdetect import detect, LangDetectException

TRAIN_PATH     = r"data\processed\split_train.jsonl"
DEV_PATH       = r"data\processed\split_dev.jsonl"
TEST_PATH      = r"data\processed\split_test.jsonl"
RETRIEVAL_PATH = r"data\processed\retrieval_corpus.jsonl"

def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def is_english(text, min_length=15):
    """
    Detect if text is English using langdetect.
    Falls back to True if text is too short to detect reliably.
    """
    if not isinstance(text, str):
        return False
    clean = re.sub(r'@\w+|https?://\S+|#\w+', '', text).strip()
    if len(clean) < min_length:
        return True  # too short to detect — keep it
    try:
        return detect(clean) == 'en'
    except LangDetectException:
        return True  # detection failed — keep it

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

# ── LANGUAGE AUDIT ───────────────────────────────────────────────
section("1. LANGUAGE AUDIT ACROSS ALL SPLITS")

for split_name, path in [
    ("Train", TRAIN_PATH),
    ("Dev",   DEV_PATH),
    ("Test",  TEST_PATH),
]:
    data = load_jsonl(path)
    total = len(data)

    # Sample 2000 for speed
    import random
    random.seed(42)
    sample = random.sample(data, min(2000, total))

    non_english = []
    for conv in sample:
        if not is_english(conv['customer_message']):
            non_english.append(conv['customer_message'])

    rate = len(non_english) / len(sample)
    estimated_total = int(rate * total)
    print(f"\n{split_name} ({total:,} total):")
    print(f"  Non-English in sample: {len(non_english)}/{len(sample)} "
          f"({rate:.1%})")
    print(f"  Estimated non-English in full split: ~{estimated_total:,}")

    if non_english:
        print(f"  Sample non-English messages:")
        for msg in non_english[:5]:
            print(f"    → {msg[:100]}")

# ── FILTER TEST SPLIT ────────────────────────────────────────────
section("2. FILTERING TEST SPLIT FOR ENGLISH ONLY")
print("(Test split is source of golden set — must be clean)")

test = load_jsonl(TEST_PATH)
test_english = [c for c in test if is_english(c['customer_message'])]
test_removed = len(test) - len(test_english)

print(f"Test before filter : {len(test):,}")
print(f"Test after filter  : {len(test_english):,}")
print(f"Removed            : {test_removed:,} ({test_removed/len(test):.1%})")

# Save filtered test
TEST_CLEAN_PATH = r"data\processed\split_test_english.jsonl"
save_jsonl(test_english, TEST_CLEAN_PATH)
print(f"Saved clean test   : {TEST_CLEAN_PATH}")

# ── NEAR-DUPLICATE DETECTION IN TEST ────────────────────────────
section("3. NEAR-DUPLICATE DETECTION IN TEST SPLIT")
print("Finding test conversations with very high BM25 score against train")
print("(Score > 50 = likely near-duplicate of a training example)")

import pickle
import numpy as np

with open(r"data\processed\bm25_index.pkl", 'rb') as f:
    bm25 = pickle.load(f)
with open(r"data\processed\bm25_corpus.pkl", 'rb') as f:
    corpus = pickle.load(f)

def tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [t for t in text.split() if len(t) > 2]

NEAR_DUP_THRESHOLD = 50.0
near_dup_flags = []

print(f"Checking {len(test_english):,} test conversations...")
for conv in test_english:
    tokens = tokenize(conv['customer_message'])
    if not tokens:
        near_dup_flags.append(False)
        continue
    scores = bm25.get_scores(tokens)
    top_score = float(scores.max())
    conv['max_bm25_vs_train'] = round(top_score, 2)
    is_near_dup = top_score > NEAR_DUP_THRESHOLD
    conv['near_duplicate_flag'] = is_near_dup
    near_dup_flags.append(is_near_dup)

n_near_dup = sum(near_dup_flags)
print(f"Near-duplicates flagged (score > {NEAR_DUP_THRESHOLD}): "
      f"{n_near_dup:,} ({n_near_dup/len(test_english):.1%})")

# Show examples of near-duplicates
near_dups = [c for c in test_english if c.get('near_duplicate_flag')]
print(f"\nSample near-duplicate test conversations:")
for c in near_dups[:5]:
    print(f"\n  Score: {c['max_bm25_vs_train']}")
    print(f"  Test  : {c['customer_message'][:120]}")

# Save flagged test
TEST_FLAGGED_PATH = r"data\processed\split_test_flagged.jsonl"
save_jsonl(test_english, TEST_FLAGGED_PATH)
print(f"\nFlagged test saved: {TEST_FLAGGED_PATH}")
print(f"(near_duplicate_flag=True conversations will be excluded "
      f"from golden set sampling)")

# ── SUMMARY ──────────────────────────────────────────────────────
section("SUMMARY")
clean_non_dup = [c for c in test_english
                 if not c.get('near_duplicate_flag')]
print(f"Test conversations available for golden set:")
print(f"  Total test         : {len(test):,}")
print(f"  After English filter: {len(test_english):,}")
print(f"  After near-dup flag : {len(clean_non_dup):,}")
print(f"  (These are safe to sample from for golden set)")

section("DONE — paste full output")