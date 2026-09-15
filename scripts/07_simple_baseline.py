"""
Phase 8 - Simple Baseline: TF-IDF + Logistic Regression
Trained on keyword-pseudo-labeled train split.
Evaluated on dev split using same keyword labels.
Will be re-evaluated on true golden labels in Phase 14.

This is Baseline 2. All numbers are OBSERVED.
"""

import json
import os
import pickle
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

TRAIN_PATH = r"data\processed\split_train.jsonl"
DEV_PATH   = r"data\processed\split_dev.jsonl"
MODEL_DIR  = r"evaluation\baselines"
SEED       = 42


def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


# ── KEYWORD LABELER (same as trivial baseline) ───────────────────
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
        'facebook login', 'google login', "can't access",
        'cannot access', "can't log", 'cannot log'
    ]),
    ('APP_TECHNICAL', [
        'update', 'version', 'crash', 'crashing', 'bug',
        'windows', 'ios', 'macos', 'slow', 'freeze', 'frozen',
        'cpu', 'install', 'reinstall', 'uninstall', 'not opening',
        "won't open", 'black screen', 'error', 'glitch'
    ]),
    ('PLAYBACK_ISSUE', [
        'stream', 'offline', 'download', 'skip',
        'stuck', 'loading', 'buffering', "won't play",
        "doesn't play", 'not playing', 'stops', 'pauses',
        'shuffle', 'repeat', 'queue', 'audio', 'volume',
        'song stops', 'music stops', 'no sound'
    ]),
    ('CONTENT_LIBRARY', [
        'album', 'artist', 'removed', 'missing', 'available',
        'playlist', 'licensed', 'region', 'library',
        'not available', 'catalogue', 'catalog', 'lyrics',
        'cover', 'release', 'add song', 'request song',
        'bring back', 'taken down'
    ]),
]

# Fixed known errors from Phase 7 analysis:
# - Removed 'play' from PLAYBACK (was matching 'playlist')
# - Removed 'android', 'iphone', 'pc', 'app' from APP_TECHNICAL
#   (too generic, cause false matches)
# - Removed 'song' from CONTENT (too short, matches too broadly)


def keyword_intent(text):
    if not isinstance(text, str):
        return 'GENERAL_ENQUIRY'
    text_lower = text.lower()
    for intent, keywords in INTENT_RULES:
        for kw in keywords:
            if kw in text_lower:
                return intent
    return 'GENERAL_ENQUIRY'


# ── LOAD & LABEL ─────────────────────────────────────────────────
section("1. LOADING AND LABELING WITH KEYWORD RULES")

def load_and_label(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            conv = json.loads(line.strip())
            conv['intent_label'] = keyword_intent(
                conv['customer_message']
            )
            data.append(conv)
    return data

train = load_and_label(TRAIN_PATH)
dev   = load_and_label(DEV_PATH)

print(f"Train size: {len(train):,}")
print(f"Dev size  : {len(dev):,}")

train_texts  = [c['customer_message'] for c in train]
train_labels = [c['intent_label'] for c in train]
dev_texts    = [c['customer_message'] for c in dev]
dev_labels   = [c['intent_label'] for c in dev]

# Label distribution
section("2. TRAINING LABEL DISTRIBUTION")
label_counts = Counter(train_labels)
total_train = len(train_labels)
for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"  {label:<25} {count:>7,}  ({count/total_train:.1%})")


# ── TRAIN TFIDF + LR ─────────────────────────────────────────────
section("3. TRAINING TF-IDF + LOGISTIC REGRESSION")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),     # unigrams and bigrams
    max_features=20_000,    # top 20k features
    min_df=2,               # ignore terms appearing < 2 times
    sublinear_tf=True,      # apply log normalization
    strip_accents='unicode',
    lowercase=True
)

print("Fitting TF-IDF vectorizer...")
X_train = vectorizer.fit_transform(train_texts)
X_dev   = vectorizer.transform(dev_texts)
print(f"Feature matrix shape (train): {X_train.shape}")
print(f"Feature matrix shape (dev):   {X_dev.shape}")

print("\nTraining Logistic Regression...")
clf = LogisticRegression(
    max_iter=1000,
    random_state=SEED,
    C=1.0,                  # regularization strength
    class_weight='balanced' # handles class imbalance
)
clf.fit(X_train, train_labels)
print("Training complete.")


# ── EVALUATE ON DEV ──────────────────────────────────────────────
section("4. EVALUATION ON DEV SET")
print("(Using keyword pseudo-labels — not true human labels)")
print("(True evaluation will happen on golden set in Phase 14)\n")

dev_preds = clf.predict(X_dev)
dev_probs = clf.predict_proba(X_dev)

acc = accuracy_score(dev_labels, dev_preds)
print(f"Accuracy : {acc:.4f} ({acc:.1%})")
print(f"\nFull classification report:")
print(classification_report(
    dev_labels, dev_preds,
    target_names=sorted(set(dev_labels))
))

print("Confusion matrix:")
labels_sorted = sorted(set(dev_labels))
cm = confusion_matrix(dev_labels, dev_preds, labels=labels_sorted)
print("Rows = actual, Cols = predicted")
print(f"{'':25}", end='')
for l in labels_sorted:
    print(f"{l[:8]:>10}", end='')
print()
for i, row_label in enumerate(labels_sorted):
    print(f"{row_label:<25}", end='')
    for val in cm[i]:
        print(f"{val:>10}", end='')
    print()


# ── CONFIDENCE DISTRIBUTION ──────────────────────────────────────
section("5. CONFIDENCE DISTRIBUTION")
max_probs = dev_probs.max(axis=1)
print(f"Mean max confidence  : {max_probs.mean():.3f}")
print(f"Median max confidence: {np.median(max_probs):.3f}")
print(f"% predictions > 0.70 : {(max_probs > 0.70).mean():.1%}")
print(f"% predictions > 0.50 : {(max_probs > 0.50).mean():.1%}")
print(f"% predictions > 0.30 : {(max_probs > 0.30).mean():.1%}")


# ── ESCALATION USING INTENT PREDICTION ──────────────────────────
section("6. SIMPLE ESCALATION USING INTENT PREDICTION")
print("Policy: if predicted intent in {SUB_BILLING, ACCOUNT_ACCESS} → ESCALATE")
print("        else → AUTO_HANDLE")
print("(Evidence: 70.9% and 70.1% real escalation rates from Phase 7)\n")

HIGH_RISK_INTENTS = {'SUB_BILLING', 'ACCOUNT_ACCESS'}
escalation_preds = [
    True if p in HIGH_RISK_INTENTS else False
    for p in dev_preds
]
actual_escalations = [c['is_dm_deflection'] for c in dev]

# Metrics
tp = sum(1 for p, a in zip(escalation_preds, actual_escalations) if p and a)
fp = sum(1 for p, a in zip(escalation_preds, actual_escalations) if p and not a)
tn = sum(1 for p, a in zip(escalation_preds, actual_escalations) if not p and not a)
fn = sum(1 for p, a in zip(escalation_preds, actual_escalations) if not p and a)

precision_esc = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_esc    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_esc        = 2 * precision_esc * recall_esc / (precision_esc + recall_esc) \
                if (precision_esc + recall_esc) > 0 else 0

false_auto_handle = fn  # said AUTO but should ESCALATE
false_auto_rate   = fn / len(dev)

coverage = sum(1 for p in escalation_preds if not p) / len(dev)

print(f"Escalation Precision : {precision_esc:.1%}")
print(f"Escalation Recall    : {recall_esc:.1%}")
print(f"Escalation F1        : {f1_esc:.1%}")
print(f"False auto-handle    : {false_auto_handle:,} ({false_auto_rate:.1%})")
print(f"  (Trivial baseline false auto-handle was: 34.7%)")
print(f"Automation coverage  : {coverage:.1%}")


# ── SAVE MODEL ───────────────────────────────────────────────────
section("7. SAVING MODEL")
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)
with open(os.path.join(MODEL_DIR, 'lr_classifier.pkl'), 'wb') as f:
    pickle.dump(clf, f)
print(f"Vectorizer saved: {MODEL_DIR}\\tfidf_vectorizer.pkl")
print(f"Classifier saved: {MODEL_DIR}\\lr_classifier.pkl")

section("DONE — paste full output")