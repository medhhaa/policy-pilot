# Smoke test
import os, sys

# make sure your project root is on PYTHONPATH
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from policy_pilot.retrieval import query_faiss
from policy_pilot.llm_agent import answer_query


def main():
    samples = [
        "Which GDPR articles govern cross‑border data transfers, and what steps must we take to comply?",
        "Our healthcare app stores patient records and geolocation data. Do we need HIPAA compliance, and which sections cover encryption and secure transmission?",
        "We're building a fintech platform in California with paid subscriptions. What CCPA user rights (e.g., data deletion, opt‑out) must we implement, and how should we handle 'Do Not Sell My Personal Information' requests?",
        "As an online retailer processing credit‑card payments, which PCI‑DSS requirements around network security and vulnerability scanning apply to us?",
        "As a publicly traded company, what Sarbanes–Oxley (SOX) Section 404 internal controls should we have in place for our financial reporting automation, and what documentation is required?",
    ]

    for q in samples:
        print("\n" + "=" * 80)
        print(f"❓ Question: {q}\n")
        # 1) Retrieve context
        ctx = query_faiss(q, top_k=3)
        for i, c in enumerate(ctx, 1):
            print(f"{i}. ({c['score']:.3f}) {c['text'][:200]}…\n")
        # 2) Generate answer
        ans = answer_query(q, ctx)
        print("🤖 Answer:\n", ans)


if __name__ == "__main__":
    main()
