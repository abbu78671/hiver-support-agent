"""
Phase 13 - Golden Set Sampling
Samples 200 conversations from clean test pool for hand-labelling.
Uses stratified sampling by predicted intent.
Golden set is LOCKED after creation — do not re-run.
"""

import json
import random
import os
import csv

FLAGGED_TEST_PATH = r"data\processed\split_test_flagged.jsonl"
GOLDEN_RAW_PATH   = r"evaluation\golden_set\golden_set_to_label.csv"
SEED = 42

def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

# ── KEYWORD LABELER (same as before) ────────────────────────────
INTENT_RULES = [
    ('SUB_BILLING', [
        'premium', 'student', 'family plan', 'charged', 'charge',
        'billing', 'trial', 'cancel', 'subscription', 'payment',
        'refund', 'price', 'discount', 'upgrade', 'downgrade',
        'free trial', 'invoice', 'cost', 'money'
    ]),
    ('ACCOUNT_ACCESS', [
        'login', 'log in', 'log-in', 'password', 'sign in',
        'sign-in', 'locked', 'forgot', 'reset', 'username',
        "can't access", 'cannot access', "can't log", 'cannot log'
    ]),
    ('APP_TECHNICAL', [
        'update', 'version', 'crash', 'crashing', 'bug',
        'windows', 'ios', 'macos', 'slow', 'freeze', 'frozen',
        'cpu', 'install', 'reinstall', 'not opening',
        "won't open", 'black screen', 'error', 'glitch'
    ]),
    ('PLAYBACK_ISSUE', [
        'stream', 'offline', 'download', 'skip', 'stuck',
        'loading', 'buffering', "won't play", "doesn't play",
        'not playing', 'stops', 'pauses', 'shuffle',
        'repeat', 'queue', 'audio', 'volume', 'no sound'
    ]),
    ('CONTENT_LIBRARY', [
        'album', 'artist', 'removed', 'missing', 'available',
        'playlist', 'licensed', 'region', 'library',
        'not available', 'catalogue', 'catalog', 'lyrics',
        'cover', 'release', 'add song', 'bring back', 'taken down'
    ]),
]

def keyword_intent(text):
    if not isinstance(text, str):
        return 'GENERAL_ENQUIRY'
    text_lower = text.lower()
    for intent, keywords in INTENT_RULES:
        for kw in keywords:
            if kw in text_lower:
                return intent
    return 'GENERAL_ENQUIRY'

# ── LOAD CLEAN TEST POOL ─────────────────────────────────────────
section("1. LOADING CLEAN TEST POOL")
pool = []
with open(FLAGGED_TEST_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            pool.append(json.loads(line))

# Keep only non-near-duplicate conversations
clean_pool = [c for c in pool if not c.get('near_duplicate_flag', False)]
print(f"Total test pool    : {len(pool):,}")
print(f"After near-dup filter: {len(clean_pool):,}")

# Add predicted intent to each conversation
for c in clean_pool:
    c['predicted_intent'] = keyword_intent(c['customer_message'])

# ── STRATIFIED SAMPLING ──────────────────────────────────────────
section("2. STRATIFIED SAMPLING")

TARGETS = {
    'PLAYBACK_ISSUE'    : 38,
    'APP_TECHNICAL'     : 35,
    'SUB_BILLING'       : 38,
    'ACCOUNT_ACCESS'    : 25,
    'CONTENT_LIBRARY'   : 25,
    'GENERAL_ENQUIRY'   : 25,
}

# Group by predicted intent
by_intent = {}
for c in clean_pool:
    intent = c['predicted_intent']
    if intent not in by_intent:
        by_intent[intent] = []
    by_intent[intent].append(c)

random.seed(SEED)
selected = []

for intent, target in TARGETS.items():
    available = by_intent.get(intent, [])
    n = min(target, len(available))
    sampled = random.sample(available, n)
    selected.extend(sampled)
    print(f"  {intent:<25}: target={target}, "
          f"available={len(available):,}, sampled={n}")

# ── ADD AMBIGUOUS CASES ──────────────────────────────────────────
section("3. ADDING DELIBERATELY AMBIGUOUS CASES")

# Ambiguous = conversations where two intents could apply
# Find by checking if message matches keywords from 2+ intents
def count_matching_intents(text):
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    matched = 0
    for intent, keywords in INTENT_RULES:
        if any(kw in text_lower for kw in keywords):
            matched += 1
    return matched

selected_ids = set(c['conversation_id'] for c in selected)
ambiguous_pool = [
    c for c in clean_pool
    if c['conversation_id'] not in selected_ids
    and count_matching_intents(c['customer_message']) >= 2
]

random.shuffle(ambiguous_pool)
ambiguous_sample = ambiguous_pool[:14]
selected.extend(ambiguous_sample)
print(f"Ambiguous cases available: {len(ambiguous_pool):,}")
print(f"Ambiguous cases added    : {len(ambiguous_sample)}")

total_selected = len(selected)
print(f"\nTotal golden set size: {total_selected}")

# ── SHUFFLE FINAL SET ────────────────────────────────────────────
random.shuffle(selected)

# ── SAVE AS CSV FOR HAND-LABELLING ──────────────────────────────
section("4. SAVING CSV FOR HAND-LABELLING")
os.makedirs('evaluation/golden_set', exist_ok=True)

fieldnames = [
    'row_id',
    'conversation_id',
    'customer_message',
    'spotify_reply',
    'predicted_intent',        # keyword prediction — DO NOT USE for labelling
    'is_dm_deflection',        # system flag — use as hint only
    # ── FIELDS YOU FILL IN ──────────────────────────────────────
    'true_intent',             # YOUR label: one of 6 intents
    'escalation_decision',     # YOUR label: AUTO_HANDLE or ESCALATE
    'escalation_reason',       # YOUR label: reason if ESCALATE, else leave blank
    'labelling_notes',         # optional: anything you noticed
]

with open(GOLDEN_RAW_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i, conv in enumerate(selected):
        writer.writerow({
            'row_id'              : i + 1,
            'conversation_id'     : conv['conversation_id'],
            'customer_message'    : conv['customer_message'],
            'spotify_reply'       : conv.get('spotify_reply', ''),
            'predicted_intent'    : conv['predicted_intent'],
            'is_dm_deflection'    : conv['is_dm_deflection'],
            'true_intent'         : '',   # YOU FILL THIS IN
            'escalation_decision' : '',   # YOU FILL THIS IN
            'escalation_reason'   : '',   # YOU FILL THIS IN IF ESCALATE
            'labelling_notes'     : '',   # optional
        })

print(f"CSV saved: {GOLDEN_RAW_PATH}")
print(f"Total rows to label: {total_selected}")

# ── PRINT LABELLING INSTRUCTIONS ─────────────────────────────────
section("5. LABELLING INSTRUCTIONS — READ BEFORE OPENING CSV")

instructions = """
GOLDEN SET LABELLING GUIDE
===========================

Open: evaluation/golden_set/golden_set_to_label.csv

For each row, fill in THREE columns:

─────────────────────────────────────────────────
COLUMN: true_intent
─────────────────────────────────────────────────
Read ONLY the customer_message column.
Ignore predicted_intent — it is often wrong.

Choose exactly ONE of:
  PLAYBACK_ISSUE    → Cannot play/stream/download music
  APP_TECHNICAL     → App bug, crash, update, device issue
  SUB_BILLING       → Premium, billing, charges, trial, cancel
  ACCOUNT_ACCESS    → Cannot log in, password, account recovery
  CONTENT_LIBRARY   → Missing song/album/artist, licensing
  GENERAL_ENQUIRY   → Feature request, feedback, praise, unclear

─────────────────────────────────────────────────
COLUMN: escalation_decision
─────────────────────────────────────────────────
Ask: "Could a bot handle this safely?"

Write AUTO_HANDLE if:
  - Issue is common and has a known self-service fix
  - No account credentials are needed
  - No money or sensitive data is involved
  - Spotify's reply gives a complete public answer

Write ESCALATE if:
  - Involves billing dispute or unexpected charge
  - Requires account lookup (email/username needed)
  - Issue is complex with no obvious public fix
  - Customer expresses urgency or distress
  - Spotify's actual reply went to DM (is_dm_deflection=True)

─────────────────────────────────────────────────
COLUMN: escalation_reason (only if ESCALATE)
─────────────────────────────────────────────────
Write a SHORT reason. Examples:
  "billing dispute requires account lookup"
  "account credentials needed"
  "complex technical issue, no public fix"
  "customer distress"
  "missing content requires internal investigation"

─────────────────────────────────────────────────
TIPS
─────────────────────────────────────────────────
- Label based on customer_message alone
- If genuinely ambiguous between two intents, pick the
  primary problem and note it in labelling_notes
- Do not change your labels after finishing
- Aim for ~200 labels in one sitting for consistency
- If you are unsure, ESCALATE is safer than AUTO_HANDLE

WHEN DONE: Save the CSV as:
  evaluation/golden_set/golden_set_labelled.csv
"""
print(instructions)

section("DONE")
print("Run this script ONCE. Do not re-run with different seeds.")
print("The golden set is now locked for sampling.")
print(f"\nNext step: open {GOLDEN_RAW_PATH} and label all {total_selected} rows.")