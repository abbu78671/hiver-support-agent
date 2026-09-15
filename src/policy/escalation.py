"""
src/policy/escalation.py
Evidence-backed escalation policy for SpotifyCares.

Escalation signals (all evidence-based from Phase 7 data):
1. High-risk intent: SUB_BILLING (70.9% real escalation rate)
                     ACCOUNT_ACCESS (70.1% real escalation rate)
2. Low intent confidence
3. Low retrieval similarity (no good historical precedent)
4. Generation failure (model returned None)

Thresholds tuned on DEV SET ONLY — never on golden set.
Current thresholds are defaults; tune with scripts/11_tune_thresholds.py
"""

HIGH_RISK_INTENTS = {'SUB_BILLING', 'ACCOUNT_ACCESS'}

# Defaults — tune these on dev set in Phase 12 tuning
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_RETRIEVAL_THRESHOLD  = 5.0


def escalation_decision(
    intent,
    intent_confidence,
    retrieval_score,
    generation_failed=False,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    retrieval_threshold=DEFAULT_RETRIEVAL_THRESHOLD,
):
    """
    Returns (decision, reason) where:
    - decision: 'ESCALATE' or 'AUTO_HANDLE'
    - reason: human-readable explanation

    Priority order (highest to lowest):
    1. Generation failure → always escalate
    2. High-risk intent → always escalate
    3. Low confidence → escalate
    4. Low retrieval → escalate
    5. Otherwise → auto-handle
    """

    if generation_failed:
        return 'ESCALATE', 'Reply generation failed — cannot auto-handle safely'

    if intent in HIGH_RISK_INTENTS:
        return 'ESCALATE', (
            f'High-risk intent ({intent}): '
            f'{70.9 if intent == "SUB_BILLING" else 70.1:.0f}% '
            f'historical escalation rate'
        )

    if intent_confidence < confidence_threshold:
        return 'ESCALATE', (
            f'Low intent confidence ({intent_confidence:.2f} < '
            f'{confidence_threshold}): ambiguous customer message'
        )

    if retrieval_score < retrieval_threshold:
        return 'ESCALATE', (
            f'Low retrieval similarity ({retrieval_score:.1f} < '
            f'{retrieval_threshold}): insufficient historical precedent'
        )

    return 'AUTO_HANDLE', (
        f'Intent={intent} (conf={intent_confidence:.2f}), '
        f'retrieval={retrieval_score:.1f} — within policy thresholds'
    )


if __name__ == "__main__":
    # Smoke test
    test_cases = [
        ("SUB_BILLING",      0.95, 18.0, False),
        ("ACCOUNT_ACCESS",   0.88, 22.0, False),
        ("PLAYBACK_ISSUE",   0.82, 16.0, False),
        ("PLAYBACK_ISSUE",   0.35, 16.0, False),
        ("APP_TECHNICAL",    0.75, 3.0,  False),
        ("CONTENT_LIBRARY",  0.70, 14.0, True),
    ]
    for intent, conf, ret_score, failed in test_cases:
        decision, reason = escalation_decision(
            intent, conf, ret_score, failed
        )
        print(f"{intent:<20} conf={conf:.2f} ret={ret_score:.1f} "
              f"→ {decision}")
        print(f"  Reason: {reason}")