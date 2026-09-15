\# Decision Log



\## 1. Selected SpotifyCares as the target brand

The dataset contains a large number of SpotifyCares customer-support interactions, giving enough examples to build and evaluate a brand-specific support system.



\## 2. Defined six support intents

I grouped incoming customer requests into six operationally meaningful intents:

PLAYBACK\_ISSUE, APP\_TECHNICAL, SUB\_BILLING, ACCOUNT\_ACCESS, CONTENT\_LIBRARY, and GENERAL\_ENQUIRY.

This keeps the classification problem manageable while covering the major support patterns in the selected data.



\## 3. Used a hand-labelled evaluation set rather than evaluating only on training data

A fixed golden evaluation set provides a consistent test set for comparing system changes.



\## 4. Used TF-IDF + Logistic Regression for the initial intent classifier

This provides a strong, inexpensive baseline for short customer-support text and is fast enough to run locally on CPU.



\## 5. Used BM25 for retrieval

BM25 performs well for lexical matching on short support messages and is lightweight, interpretable, and easy to run without GPU infrastructure.



\## 6. Retrieve historical support cases before generation

The generator receives relevant historical support examples as evidence so that responses are grounded in previously observed support patterns instead of relying only on the language model's prior knowledge.



\## 7. Selected Phi-3 Mini for local generation

Phi-3 Mini provides a reasonable trade-off between response quality, inference cost, and local hardware requirements.



\## 8. Used Ollama for local inference

Running the model locally avoids API costs and makes the system reproducible in a zero-cost development environment.



\## 9. Added an escalation policy

The system should not automatically answer every request. Sensitive, ambiguous, low-confidence, or poorly supported cases should be escalated instead of risking an incorrect automated response.



\## 10. Consider classifier confidence in escalation

A low-confidence intent prediction is a useful risk signal because uncertain classification can lead to inappropriate retrieval and response generation.



\## 11. Consider retrieval quality in escalation

Even when the intent is classified correctly, weak retrieval evidence can make generated responses unreliable. Therefore poor retrieval can trigger escalation.



\## 12. Included a trivial baseline

The majority-class baseline establishes the minimum performance expected from a system with no meaningful classification capability.



\## 13. Included a simple ML baseline

TF-IDF + Logistic Regression provides a second reference point for determining whether the final support-agent pipeline adds value beyond straightforward supervised text classification.



\## 14. Prioritized measurable failure modes over a larger feature set

The assignment emphasizes evaluation and failure analysis, so I prioritized reliable metrics, reproducibility, and error inspection instead of adding unnecessary system complexity.



\## 15. Chose not to build a fully autonomous multi-agent system

The assignment focuses on customer-support automation, retrieval, generation, evaluation, and safe escalation. A multi-agent architecture would add complexity without being necessary to demonstrate these capabilities.

