"""
Tests for escalation policy.
Run: python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.policy.escalation import escalation_decision


def test_sub_billing_always_escalates():
    decision, reason = escalation_decision(
        intent='SUB_BILLING',
        intent_confidence=0.99,
        retrieval_score=25.0,
        generation_failed=False
    )
    assert decision == 'ESCALATE'
    assert 'SUB_BILLING' in reason


def test_account_access_always_escalates():
    decision, reason = escalation_decision(
        intent='ACCOUNT_ACCESS',
        intent_confidence=0.99,
        retrieval_score=25.0,
        generation_failed=False
    )
    assert decision == 'ESCALATE'
    assert 'ACCOUNT_ACCESS' in reason


def test_low_confidence_escalates():
    decision, reason = escalation_decision(
        intent='PLAYBACK_ISSUE',
        intent_confidence=0.30,
        retrieval_score=15.0,
        generation_failed=False
    )
    assert decision == 'ESCALATE'
    assert 'confidence' in reason.lower()


def test_low_retrieval_escalates():
    decision, reason = escalation_decision(
        intent='PLAYBACK_ISSUE',
        intent_confidence=0.85,
        retrieval_score=2.0,
        generation_failed=False
    )
    assert decision == 'ESCALATE'
    assert 'retrieval' in reason.lower()


def test_generation_failure_escalates():
    decision, reason = escalation_decision(
        intent='CONTENT_LIBRARY',
        intent_confidence=0.80,
        retrieval_score=15.0,
        generation_failed=True
    )
    assert decision == 'ESCALATE'
    assert 'generation' in reason.lower()


def test_good_signal_auto_handles():
    decision, reason = escalation_decision(
        intent='PLAYBACK_ISSUE',
        intent_confidence=0.85,
        retrieval_score=18.0,
        generation_failed=False
    )
    assert decision == 'AUTO_HANDLE'


def test_escalation_reason_never_empty():
    for intent in ['SUB_BILLING', 'ACCOUNT_ACCESS',
                   'PLAYBACK_ISSUE', 'APP_TECHNICAL']:
        _, reason = escalation_decision(
            intent=intent,
            intent_confidence=0.75,
            retrieval_score=15.0
        )
        assert reason
        assert len(reason) > 5