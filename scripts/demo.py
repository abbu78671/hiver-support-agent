"""
Demo script - test the system on any customer message.
Usage: python scripts\demo.py
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.generator import load_retrieval_components, generate_reply
from src.policy.escalation import escalation_decision


def section(title):
    print("\n" + "="*55)
    print(f"  {title}")
    print("="*55)


def run(customer_message):
    print(f"\n{'─'*55}")
    print(f"CUSTOMER MESSAGE:")
    print(f"  {customer_message}")
    print(f"{'─'*55}")

    # Step 1 — Intent Classification
    with open('evaluation/baselines/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('evaluation/baselines/lr_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)

    X = vectorizer.transform([customer_message])
    pred_intent = clf.predict(X)[0]
    pred_probs  = clf.predict_proba(X)[0]
    confidence  = float(pred_probs.max())

    print(f"\n[1] INTENT CLASSIFICATION")
    print(f"    Predicted : {pred_intent}")
    print(f"    Confidence: {confidence:.1%}")

    # Step 2 — Retrieval
    bm25, corpus = load_retrieval_components()
    from src.generation.generator import retrieve
    retrieved = retrieve(customer_message, bm25, corpus, top_k=3)

    top_score = retrieved[0]['bm25_score'] if retrieved else 0.0
    print(f"\n[2] RETRIEVAL (top {len(retrieved)} historical cases)")
    for i, r in enumerate(retrieved):
        print(f"    [{i+1}] score={r['bm25_score']:.1f} | "
              f"{r['customer_message'][:60]}...")

    # Step 3 — Escalation Decision (before generation)
    decision, reason = escalation_decision(
        intent=pred_intent,
        intent_confidence=confidence,
        retrieval_score=top_score,
        generation_failed=False,
    )

    print(f"\n[3] ESCALATION DECISION")
    print(f"    Decision : {decision}")
    print(f"    Reason   : {reason}")

    # Step 4 — Generate reply (only if AUTO_HANDLE)
    if decision == 'AUTO_HANDLE':
        print(f"\n[4] GENERATING REPLY (this takes ~30 seconds)...")
        result = generate_reply(
            customer_message, pred_intent, bm25, corpus
        )
        reply = result.get('generated_reply', 'Generation failed')
        print(f"\n    DRAFT REPLY:")
        print(f"    {reply}")
        print(f"\n    Latency: {result['latency_seconds']}s")
    else:
        print(f"\n[4] REPLY GENERATION SKIPPED")
        print(f"    This case escalates to a human agent.")
        print(f"    No reply generated.")

    print(f"\n{'─'*55}")
    print(f"FINAL OUTPUT:")
    print(f"  Decision : {decision}")
    if decision == 'AUTO_HANDLE':
        print(f"  Reply    : {reply}")
    else:
        print(f"  Reason   : {reason}")
    print(f"{'─'*55}\n")


# ── TEST MESSAGES ─────────────────────────────────────────────────
if __name__ == "__main__":

    test_messages = [
        "My Spotify app keeps crashing when I try to open it on iPhone",
        "I was charged twice for premium this month, please help",
        "Why has the album I saved disappeared from my library?",
        "I cant log into my account, forgot my password",
        "When will you add Taylor Swift's new album to Spotify?",
    ]

    print("SPOTIFYCARES AI SUPPORT AGENT — LIVE DEMO")
    print("==========================================")
    print("Make sure ollama is running: ollama serve")

    for msg in test_messages:
        run(msg)
        input("\nPress ENTER for next message...")