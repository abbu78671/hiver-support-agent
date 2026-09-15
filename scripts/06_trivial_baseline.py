"""
Phase 7 - Trivial Baseline
Two trivial baselines:
  A) Intent: keyword-rule labeler
  B) Escalation: always predict AUTO-HANDLE

These set the performance floor.
All numbers are OBSERVED from actual runs.
"""

import json
import re
from collections import Counter, defaultdict

DEV_PATH = r"data\processed\split_dev.jsonl"
OUTPUT_PATH = r"evaluation\baselines\trivial_baseline_results.json"

# ── TAXONOMY ─────────────────────────────────────────────────────
# ORDER MATTERS: check SUB_BILLING before ACCOUNT
# because "can't access premium" should be SUB_BILLING not ACCOUNT

INTENT_RULES = [
    (
        'SUB_BILLING',
        ['premium', 'student', 'family plan', 'charged', 'charge',
         'billing', 'trial', 'cancel', 'subscription', 'payment',
         'refund', 'price', 'discount', 'plan', 'upgrade',
         'downgrade', 'free trial', 'invoice', 'cost', 'money']
    ),
    (
        'ACCOUNT_ACCESS',
        ['login', 'log in', 'log-in', 'password', 'sign in',
         'sign-in', 'locked', 'forgot', 'reset', 'username',
         'facebook login', 'google login', 'can\'t access',
         'cannot access', 'account recovery', 'verify email',
         'email address', 'can\'t log', 'cannot log']
    ),
    (
        'APP_TECHNICAL',
        ['update', 'version', 'crash', 'crashing', 'bug',
         'iphone', 'android', 'windows', 'ios', 'macos',
         'pc ', ' app ', 'slow', 'freeze', 'frozen', 'cpu',
         'install', 'reinstall', 'uninstall', 'not opening',
         'won\'t open', 'black screen', 'error', 'glitch']
    ),
    (
        'PLAYBACK_ISSUE',
        ['play', 'stream', 'offline', 'download', 'skip',
         'stuck', 'loading', 'buffering', 'won\'t play',
         'doesn\'t play', 'not playing', 'stops', 'pauses',
         'shuffle', 'repeat', 'queue', 'audio', 'sound',
         'volume', 'song stops', 'music stops', 'no sound']
    ),
    (
        'CONTENT_LIBRARY',
        ['album', 'artist', 'removed', 'missing', 'available',
         'playlist', 'licensed', 'region', 'library', 'track',
         'song not', 'not available', 'catalogue', 'catalog',
         'content', 'lyrics', 'cover', 'release', 'add song',
         'request song', 'bring back', 'taken down']
    ),
]
# Anything that matches nothing → GENERAL_ENQUIRY


def keyword_intent(text):
    """
    Assign intent using keyword rules.
    Returns (intent_label, matched_keyword_or_None)
    """
    if not isinstance(text, str):
        return 'GENERAL_ENQUIRY', None

    text_lower = text.lower()

    for intent, keywords in INTENT_RULES:
        for kw in keywords:
            if kw in text_lower:
                return intent, kw

    return 'GENERAL_ENQUIRY', None


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ── LOAD DEV SET ─────────────────────────────────────────────────
print("Loading dev set...")
dev = []
with open(DEV_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        dev.append(json.loads(line.strip()))
print(f"Dev set size: {len(dev):,}")


# ─────────────────────────────────────────────────────────────────
# BASELINE A: INTENT — KEYWORD RULES
# ─────────────────────────────────────────────────────────────────
section("BASELINE A — INTENT: KEYWORD RULES")

intent_predictions = []
matched_keywords = []

for conv in dev:
    pred, kw = keyword_intent(conv['customer_message'])
    intent_predictions.append(pred)
    matched_keywords.append(kw)
    conv['predicted_intent'] = pred

# Intent distribution
intent_counts = Counter(intent_predictions)
total = len(intent_predictions)

print("\nPredicted intent distribution on dev set:")
for intent, count in sorted(intent_counts.items(),
                             key=lambda x: -x[1]):
    print(f"  {intent:<25} {count:>6,}  ({count/total:.1%})")

# Majority class
majority_intent = intent_counts.most_common(1)[0][0]
majority_count = intent_counts.most_common(1)[0][1]
print(f"\nMajority class: {majority_intent} ({majority_count/total:.1%})")
print("(Majority-class baseline for intent = always predict this)")
print(f"Majority-class accuracy would be: {majority_count/total:.1%}")

# Coverage: how many messages matched any keyword?
matched = sum(1 for p in intent_predictions if p != 'GENERAL_ENQUIRY')
print(f"\nMessages matched by keyword rules: {matched:,} ({matched/total:.1%})")
print(f"Fell through to GENERAL_ENQUIRY:   "
      f"{total-matched:,} ({(total-matched)/total:.1%})")

# Sample 5 from each intent for sanity check
section("SAMPLE PREDICTIONS PER INTENT (sanity check)")
by_intent = defaultdict(list)
for conv in dev:
    by_intent[conv['predicted_intent']].append(conv)

for intent in ['SUB_BILLING', 'ACCOUNT_ACCESS', 'APP_TECHNICAL',
               'PLAYBACK_ISSUE', 'CONTENT_LIBRARY', 'GENERAL_ENQUIRY']:
    examples = by_intent[intent][:3]
    print(f"\n  {intent} ({len(by_intent[intent]):,} predictions):")
    for ex in examples:
        print(f"    → {ex['customer_message'][:120]}")


# ─────────────────────────────────────────────────────────────────
# BASELINE B: ESCALATION — ALWAYS AUTO-HANDLE
# ─────────────────────────────────────────────────────────────────
section("BASELINE B — ESCALATION: ALWAYS AUTO-HANDLE")

# Ground truth: is_dm_deflection == True → should ESCALATE
# is_dm_deflection == False → should AUTO-HANDLE
actual_escalate = [c['is_dm_deflection'] for c in dev]
n_should_escalate = sum(actual_escalate)
n_should_autohandle = len(actual_escalate) - n_should_escalate

print(f"\nGround truth on dev set:")
print(f"  Should ESCALATE    : {n_should_escalate:,} ({n_should_escalate/total:.1%})")
print(f"  Should AUTO-HANDLE : {n_should_autohandle:,} ({n_should_autohandle/total:.1%})")

# Trivial baseline: always predict AUTO-HANDLE
always_auto_correct = n_should_autohandle
always_auto_accuracy = always_auto_correct / total

# False auto-handle: cases we predicted AUTO but should ESCALATE
false_auto_handle = n_should_escalate  # we predicted AUTO for everything

print(f"\nTrivial baseline — ALWAYS predict AUTO-HANDLE:")
print(f"  Accuracy           : {always_auto_accuracy:.1%}")
print(f"  False auto-handles : {false_auto_handle:,} ({false_auto_handle/total:.1%})")
print(f"  (False auto-handle = customer needed human, got bot response)")
print(f"  Precision (auto)   : {n_should_autohandle/total:.1%}")
print(f"  Recall (auto)      : 100.0% (trivially — we always predict auto)")
print(f"  Recall (escalate)  : 0.0% (we never predict escalate)")

# Trivial baseline: always predict ESCALATE  
always_esc_correct = n_should_escalate
always_esc_accuracy = always_esc_correct / total
print(f"\nAlternative trivial baseline — ALWAYS predict ESCALATE:")
print(f"  Accuracy           : {always_esc_accuracy:.1%}")
print(f"  False escalations  : {n_should_autohandle:,} ({n_should_autohandle/total:.1%})")
print(f"  (False escalation = bot could handle it, wasted human time)")


# ─────────────────────────────────────────────────────────────────
# INTENT × ESCALATION CROSS-TAB
# ─────────────────────────────────────────────────────────────────
section("INTENT × ESCALATION CROSS-TAB")
print("(Shows which intents most often lead to DM deflection)")
print(f"\n{'Intent':<25} {'Total':>7} {'Escalate':>9} {'EscRate':>8}")
print("-" * 55)

for intent in ['SUB_BILLING', 'ACCOUNT_ACCESS', 'APP_TECHNICAL',
               'PLAYBACK_ISSUE', 'CONTENT_LIBRARY', 'GENERAL_ENQUIRY']:
    convs = by_intent[intent]
    if not convs:
        continue
    n = len(convs)
    n_esc = sum(1 for c in convs if c['is_dm_deflection'])
    print(f"{intent:<25} {n:>7,} {n_esc:>9,} {n_esc/n:>8.1%}")


# ─────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────
section("SAVING RESULTS")

import os
os.makedirs('evaluation/baselines', exist_ok=True)

results = {
    'baseline': 'trivial',
    'dev_set_size': total,
    'intent_distribution': dict(intent_counts),
    'majority_class': majority_intent,
    'majority_class_accuracy': round(majority_count / total, 4),
    'keyword_coverage': round(matched / total, 4),
    'escalation_always_auto': {
        'accuracy': round(always_auto_accuracy, 4),
        'false_auto_handle_count': false_auto_handle,
        'false_auto_handle_rate': round(false_auto_handle / total, 4),
    },
    'escalation_always_escalate': {
        'accuracy': round(always_esc_accuracy, 4),
    }
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {OUTPUT_PATH}")

section("DONE — paste full output")