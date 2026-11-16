"""Prompt templates for Phase 2. Placeholders <<name>> are filled with src.alignment.prompts.fill()."""

QUESTION_RULES = """Question types:
- differential: the answer hinges on a DIFFERENCE between the documents (who offers more,
  how the rules differ, by how much).
- equivalence: the answer hinges on the documents saying the SAME thing (possibly in different
  words). Good ones tempt the reader to assume a difference.
- asymmetric: one document addresses something the other does NOT (or covers it in materially
  less detail). The correct answer must say what is missing and where.
Reasoning styles: multi_hop (combine 2+ facts), conditional (a concrete employee scenario),
numeric (compute a gap, total or ratio from stated values only), direct (one attribute, still
comparing both documents).

Hard rules for the QUESTION text:
- It must need information from BOTH documents.
- Never name or hint at which document is which: no country, region, city, company-entity
  names, no jurisdiction-specific law names, no "Document A/B", "first/second document",
  "R1", page/section/clause numbers, chunk IDs or file names. Refer neutrally:
  "the two handbooks", "both policies", "each handbook", "one of the handbooks".
- Natural phrasing an employee or HR analyst would use; self-contained; exactly one correct answer.
- No bare yes/no questions - ask for the specifics as well.

Hard rules for the ANSWER fields:
- ground_truth: complete and precise, using ONLY the facts given. Name the documents as
  "<<label_a>>" and "<<label_b>>". State each document's position, then the comparison
  conclusion. For an absent item say explicitly that the document does not address it.
- short_answer: one-sentence conclusion.
- key_points: 2-6 atomic points a correct answer must contain (each document's position + conclusion).
- pair_ids: ALL fact IDs the answer relies on.
- Use only stated numbers; never assume unstated values (e.g. do not add leave types together
  unless the question explicitly asks for that total).
"""

QUESTION_SCHEMA = """{"question": "", "question_type": "differential|equivalence|asymmetric",
 "reasoning": "direct|multi_hop|conditional|numeric", "difficulty": "easy|medium|hard",
 "pair_ids": [], "ground_truth": "", "short_answer": "", "key_points": []}"""


GENERATE_TOPIC = """You are writing a golden evaluation set that tests how well an AI system COMPARES
two policy documents: "<<label_a>>" and "<<label_b>>".

Topic: <<topic>>
Verified comparison facts (ID | relation | attribute, then each document's position):
<<pairs>>

Write <<n>> questions with this type mix: <<mix>>.
At most <<max_direct>> may use the "direct" reasoning style; make the rest multi_hop,
conditional or numeric, and make them genuinely challenging.

""" + QUESTION_RULES + """
Return JSON only: {"questions": [""" + QUESTION_SCHEMA + """]}
"""


GENERATE_CROSS = """You are writing a golden evaluation set that tests how well an AI system COMPARES
two policy documents: "<<label_a>>" and "<<label_b>>".

Verified comparison facts across all topics (ID | relation | attribute, then each document's position):
<<pairs>>

Write <<n>> CROSS-TOPIC questions. The facts come from exactly two topics (see the
"### Topic" headings). EVERY question MUST cite at least one pair_id from EACH of the two
topics and combine them in one employee scenario or comparison (e.g. how probation status
affects both sick pay and notice). A question whose pair_ids all come from one topic is
invalid. Use one of these question types: <<mix>>. Avoid the "direct" reasoning style.

""" + QUESTION_RULES + """
Return JSON only: {"questions": [""" + QUESTION_SCHEMA + """]}
"""


REVISE = """You wrote this question for a golden evaluation set comparing two policy documents
("<<label_a>>" and "<<label_b>>"), but a review rejected it.

Rejected question (JSON):
<<question>>

Problems found:
<<issues>>

Verified comparison facts you may use:
<<pairs>>

Rewrite it to fix every problem. Keep the same question_type (<<qtype>>) unless the problems say
the type is wrong. """ + QUESTION_RULES + """
Return JSON only: {"question": """ + QUESTION_SCHEMA + """}
"""


CRITIC = """You are a strict reviewer of a golden evaluation dataset that tests comparison of two
policy documents ("<<label_a>>" and "<<label_b>>").

Question: <<question>>
Declared question type: <<qtype>>
  (differential = hinges on a difference; equivalence = hinges on both saying the same;
   asymmetric = one document addresses something the other does not / covers it in far less detail)
Ground truth answer: <<ground_truth>>
Key points: <<key_points>>

Evidence - the ONLY information available:
<<evidence>>

Assess:
- answerable (1-5): can the question be fully answered from this evidence alone?
- correct (1-5): is the ground truth fully correct and complete per the evidence (numbers,
  which document says what, nothing invented, no unstated assumptions)?
- unambiguous (1-5): is there exactly one clear correct answer?
- needs_both_documents (true/false): does answering really need both documents?
- type_matches (true/false): does the declared type fit the question?
- leaks_document_identity (true/false): does the QUESTION text name or hint which document
  is which (country, region, city, entity name, jurisdiction-specific law, "Document A",
  "first document", page/section numbers)?
- too_trivial (true/false): answerable by a single keyword lookup with no real comparison?
- difficulty: easy | medium | hard

Return JSON only:
{"answerable": 0, "correct": 0, "unambiguous": 0, "needs_both_documents": true,
 "type_matches": true, "leaks_document_identity": false, "too_trivial": false,
 "difficulty": "medium", "issues": ["<each concrete problem>"]}
"""
