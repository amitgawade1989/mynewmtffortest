"""LLM-judge prompts for the hybrid metrics (framework 'Dimension Deep-Dives' rubrics)."""

SEMANTIC = """You are grading an answer that compares two policy documents.

Question: <<question>>
Reference answer: <<reference>>
Model answer: <<answer>>

Judge whether the model answer reaches the same CONCLUSION as the reference, ignoring wording.
Check explicitly for:
- polarity / direction flips (e.g. says the wrong document is more generous, or "same" vs "different");
- attribution swaps (a fact from one document attributed to the other document).
Return JSON only:
{"verdict": "correct|partially_correct|incorrect", "polarity_flip": false, "attribution_swap": false,
 "reason": "<one sentence>"}
"""

RELEVANCE = """Question: <<question>>
Model answer: <<answer>>

Rate 1-5 how substantively the answer addresses what was asked (5 = fully on the ask,
1 = off-topic or tangential). IGNORE whether the answer is factually correct.
Return JSON only: {"score": 1, "reason": "<one sentence>"}
"""

COMPLETENESS = """A model answered a question that compares two policy documents.
Below are the required points (from the reference answer), each paired with the sentence of the
model answer that matched it best by embedding similarity.

Question: <<question>>
Full model answer: <<answer>>

Required points with best-matching answer sentence:
<<pairs>>

For EACH required point, decide whether the model answer (anywhere, not only the matched
sentence) states it and states it correctly.
Return JSON only: {"points": [{"index": 1, "present_and_correct": true, "note": ""}]}
"""

COHERENCE = """Model answer:
<<answer>>

Possible contradictions flagged by an automatic NLI check (may be false alarms):
<<flags>>

Score 1-5 whether the ideas follow a clear, logical sequence with no self-contradiction or
unnecessary repetition. If the score is below 4, quote the specific contradictory or repeated
sentence. Return JSON only: {"score": 1, "quote": "", "reason": "<one sentence>"}
"""

CLARITY = """Model answer:
<<answer>>

Score 1-5 how easy the writing is to understand (independent of grammar or factual correctness).
Return JSON only: {"score": 1, "reason": "<one sentence>"}
"""
