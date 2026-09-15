"""
Phase 6 - Dataset splitting
Splits at conversation level to prevent leakage.
Fixed seed = 42 everywhere.
"""

import json
import random
import os

INPUT_PATH = r"data\processed\spotify_conversations.jsonl"
TRAIN_PATH = r"data\processed\split_train.jsonl"
DEV_PATH   = r"data\processed\split_dev.jsonl"
TEST_PATH  = r"data\processed\split_test.jsonl"

SEED = 42
TRAIN_RATIO = 0.70
DEV_RATIO   = 0.15
# TEST = remaining 0.15

def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

# ── LOAD ─────────────────────────────────────────────────────────
print("Loading conversations...")
conversations = []
with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        conversations.append(json.loads(line.strip()))
print(f"Total conversations loaded: {len(conversations):,}")

# ── SHUFFLE ──────────────────────────────────────────────────────
random.seed(SEED)
random.shuffle(conversations)

# ── SPLIT ────────────────────────────────────────────────────────
n = len(conversations)
train_end = int(n * TRAIN_RATIO)
dev_end   = int(n * (TRAIN_RATIO + DEV_RATIO))

train = conversations[:train_end]
dev   = conversations[train_end:dev_end]
test  = conversations[dev_end:]

section("SPLIT SIZES")
print(f"Train : {len(train):,} ({len(train)/n:.1%})")
print(f"Dev   : {len(dev):,}   ({len(dev)/n:.1%})")
print(f"Test  : {len(test):,}  ({len(test)/n:.1%})")
print(f"Total : {n:,}")

# ── VERIFY NO OVERLAP ─────────────────────────────────────────────
section("LEAKAGE CHECK")
train_ids = set(c['conversation_id'] for c in train)
dev_ids   = set(c['conversation_id'] for c in dev)
test_ids  = set(c['conversation_id'] for c in test)

train_dev_overlap = train_ids & dev_ids
train_test_overlap = train_ids & test_ids
dev_test_overlap  = dev_ids & test_ids

print(f"Train ∩ Dev overlap  : {len(train_dev_overlap)} (must be 0)")
print(f"Train ∩ Test overlap : {len(train_test_overlap)} (must be 0)")
print(f"Dev ∩ Test overlap   : {len(dev_test_overlap)} (must be 0)")

if any([train_dev_overlap, train_test_overlap, dev_test_overlap]):
    print("LEAKAGE DETECTED — do not proceed")
else:
    print("No overlap — clean split confirmed")

# ── DEFLECTION RATES PER SPLIT ───────────────────────────────────
section("DEFLECTION RATE PER SPLIT (should be similar)")
for name, split in [("Train", train), ("Dev", dev), ("Test", test)]:
    dm = sum(1 for c in split if c['is_dm_deflection'])
    print(f"{name}: {dm/len(split):.1%} DM deflection rate")

# ── SAVE ─────────────────────────────────────────────────────────
section("SAVING SPLITS")
os.makedirs('data/processed', exist_ok=True)

for path, split in [(TRAIN_PATH, train), (DEV_PATH, dev), (TEST_PATH, test)]:
    with open(path, 'w', encoding='utf-8') as f:
        for conv in split:
            f.write(json.dumps(conv) + '\n')
    print(f"Saved: {path} ({len(split):,} conversations)")

# ── RETRIEVAL CORPUS (train, non-deflection only) ────────────────
section("RETRIEVAL CORPUS")
retrieval = [c for c in train if not c['is_dm_deflection']]
RETRIEVAL_PATH = r"data\processed\retrieval_corpus.jsonl"
with open(RETRIEVAL_PATH, 'w', encoding='utf-8') as f:
    for conv in retrieval:
        f.write(json.dumps(conv) + '\n')
print(f"Retrieval corpus: {len(retrieval):,} non-deflection train conversations")
print(f"Saved: {RETRIEVAL_PATH}")

section("DONE")
print("Splits are locked. Do not re-run with a different seed.")
print("Golden set will be drawn from test split only.")