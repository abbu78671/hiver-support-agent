"""
Phase 15+16 - LLM-as-judge for response quality.
DISCLOSURE: Judge is phi3:mini. Labels are LLM pseudo-labels.
Agreement measures LLM self-agreement, not human-LLM agreement.
"""

import json
import subprocess
import random
import os
import re
import statistics

RESULTS_PATH = r"evaluation\reports\golden_set_results.jsonl"
JUDGE_PATH   = r"evaluation\reports\judge_results.json"
OLLAMA_MODEL = "phi3:mini"
SAMPLE_SIZE  = 30
SEED         = 42


def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def call_ollama_judge(prompt):
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 80, "temperature": 0.0}
    })
    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST",
             "http://localhost:11434/api/generate",
             "-H", "Content-Type: application/json",
             "-d", payload],
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=90
        )
        response = json.loads(result.stdout)
        return response.get('response', '').strip()
    except Exception:
        return None


def extract_score(raw_text):
    """
    Robustly extract a score 1-5 from judge response.
    phi3:mini does not always follow format instructions exactly.
    We look for any digit 1-5 in the response.
    """
    if not raw_text:
        return None
    # Look for patterns like "4/5", "Score: 4", "4 out of 5", or just "4"
    patterns = [
        r'\b([1-5])\s*/\s*5',       # "4/5"
        r'score[:\s]+([1-5])',       # "Score: 4"
        r'rating[:\s]+([1-5])',      # "Rating: 4"
        r'\b([1-5])\s+out\s+of',    # "4 out of"
        r'^([1-5])\b',              # starts with digit
        r'\b([1-5])\b',             # any standalone digit
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def judge_reply(customer_message, generated_reply):
    """
    Ask phi3:mini to rate a reply on a single 1-5 scale.
    Simpler prompt = more reliable score extraction.
    """
    prompt = (
        f"Rate this customer support reply from 1 to 5.\n"
        f"1=very poor, 2=poor, 3=acceptable, 4=good, 5=excellent\n\n"
        f"Customer said: {customer_message[:150]}\n"
        f"Support replied: {generated_reply[:200]}\n\n"
        f"Reply with a single number from 1 to 5 only. "
        f"No explanation needed. Your rating:"
    )
    raw = call_ollama_judge(prompt)
    score = extract_score(raw)
    return score, raw


# ── LOAD RESULTS ─────────────────────────────────────────────────
section("1. LOADING EVALUATION RESULTS")
results = []
with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

auto_handled = [
    r for r in results
    if not r['generation_failed']
    and r['pred_escalation'] == 'AUTO_HANDLE'
    and r.get('generated_reply', '').strip()
]
print(f"Total results         : {len(results)}")
print(f"Auto-handled with reply: {len(auto_handled)}")

random.seed(SEED)
sample = random.sample(auto_handled, min(SAMPLE_SIZE, len(auto_handled)))
print(f"Sample for judging    : {len(sample)}")

# ── JUDGE ────────────────────────────────────────────────────────
section("2. JUDGING REPLIES")
print("Single 1-5 overall score per reply.\n")

judge_results = []

for i, r in enumerate(sample):
    score, raw = judge_reply(
        r['customer_message'],
        r.get('generated_reply', '')
    )

    judge_results.append({
        'row_id':      r['row_id'],
        'message':     r['customer_message'][:100],
        'reply':       r.get('generated_reply', '')[:150],
        'true_intent': r['true_intent'],
        'score':       score,
        'raw_judge':   raw,
    })

    status = str(score) if score else "FAILED"
    if (i + 1) % 5 == 0:
        print(f"  [{i+1:>2}/{len(sample)}] last score={status}")

# ── RESULTS ──────────────────────────────────────────────────────
section("3. JUDGE SCORES")

valid   = [j for j in judge_results if j['score'] is not None]
failed  = [j for j in judge_results if j['score'] is None]
scores  = [j['score'] for j in valid]

print(f"Successfully scored: {len(valid)}/{len(sample)}")
print(f"Failed to score   : {len(failed)}/{len(sample)}")

if failed:
    print("\nFailed extractions (raw judge output):")
    for j in failed[:3]:
        print(f"  raw: {repr(j['raw_judge'])}")

if scores:
    mean_score   = statistics.mean(scores)
    median_score = statistics.median(scores)
    print(f"\nOverall quality score (1-5 scale):")
    print(f"  Mean   : {mean_score:.2f}")
    print(f"  Median : {median_score:.1f}")
    print(f"  Min    : {min(scores)}")
    print(f"  Max    : {max(scores)}")

    print(f"\nScore distribution:")
    for s in [1, 2, 3, 4, 5]:
        count = scores.count(s)
        bar   = '█' * count
        print(f"  {s}: {bar} ({count})")

    # Per-intent breakdown
    print(f"\nMean score by intent:")
    intents = ['PLAYBACK_ISSUE', 'APP_TECHNICAL', 'CONTENT_LIBRARY',
               'GENERAL_ENQUIRY', 'SUB_BILLING', 'ACCOUNT_ACCESS']
    for intent in intents:
        intent_scores = [j['score'] for j in valid
                         if j['true_intent'] == intent]
        if intent_scores:
            print(f"  {intent:<25}: {statistics.mean(intent_scores):.2f} "
                  f"(n={len(intent_scores)})")

# ── DISCLOSURE ───────────────────────────────────────────────────
section("4. DISCLOSURE")
print(
    "Judge model : phi3:mini\n"
    "Label source: Claude Sonnet 4.6 (LLM pseudo-labels)\n"
    "This measures LLM self-agreement, NOT human-LLM agreement.\n"
    "True human evaluation was not performed (time constraint).\n"
    "This limitation must be stated explicitly in the report."
)

# ── SAVE ─────────────────────────────────────────────────────────
section("5. SAVING")
os.makedirs('evaluation/reports', exist_ok=True)

output = {
    'disclosure': (
        'Judge is phi3:mini. Labels are LLM pseudo-labels. '
        'Scores reflect LLM self-agreement, not human judgment.'
    ),
    'sample_size':   len(sample),
    'valid_scores':  len(valid),
    'failed_scores': len(failed),
    'mean_score':    round(statistics.mean(scores), 2) if scores else None,
    'median_score':  statistics.median(scores) if scores else None,
    'individual':    judge_results,
}

with open(JUDGE_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)
print(f"Saved: {JUDGE_PATH}")

section("DONE — paste full output")