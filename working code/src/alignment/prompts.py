"""Prompt templates for Phase 1. Placeholders look like <<name>> and are filled with `fill()`."""


def fill(template, **values):
    """Replace <<name>> placeholders (avoids clashes with JSON braces in the templates)."""
    for k, v in values.items():
        template = template.replace(f"<<{k}>>", str(v))
    return template


FACT_EXTRACTION = """You are building a fact inventory from ONE section of a policy document.

Section title: <<title>>
Chunks (each starts with its chunk ID in square brackets):
<<chunks>>

Extract EVERY concrete, checkable fact that could be compared against another organisation's
version of this kind of document: entitlements, amounts, durations, limits, thresholds,
eligibility conditions, procedures and their steps, deadlines, laws/frameworks referenced,
responsible roles, exceptions, document metadata (dates, versions).

Rules:
- One fact per attribute. "attribute" is a short GENERIC name that would also fit another
  organisation's document (e.g. "Annual leave entitlement", "Notice period after probation").
  Never put company names, countries, regions or section numbers in "attribute".
- "value": precise content from the text, including numbers, units and conditions. Do not infer
  anything that is not stated. If a sentence is cut off, use only its complete part.
- "child_ids": the chunk ID(s) the fact comes from.
- "quote": an EXACT copy (max 30 words) from that chunk that supports the value.
- "category": a broad topic label (e.g. "Leave", "Working hours", "Pay & benefits", "Termination").
- Do NOT extract section headings or titles as facts.

Return JSON only:
{"facts": [{"attribute": "", "value": "", "category": "", "child_ids": [""], "quote": ""}]}
"""


SECTION_MATCH = """Two policy documents (Document A and Document B) are being compared topic by topic.
Below is one section from Document A and candidate sections from Document B.
Decide which candidates cover ANY of the same subject matter as the Document A section,
even if the heading is different or the overlap is only partial.

Document A section [<<parent_id>>] "<<title>>":
<<text>>

Candidate sections from Document B:
<<candidates>>

Return JSON only:
{"topic_name": "<short generic name for the subject of the Document A section>",
 "matches": [{"parent_id": "<candidate ID>", "overlap": "full|partial", "shared_topics": ["..."]}]}
Use an empty "matches" list if no candidate overlaps.
"""


FACT_PAIRING = """Compare two policy documents on the topic "<<topic>>".

Facts from Document A:
<<facts_a>>

Facts from Document B:
<<facts_b>>

Group the facts that address the same attribute / question, and classify each group:
- SAME: both documents state the same rule or value (wording may differ).
- DIFFERENT: both address it but the values or rules differ.
- PARTIAL: both address it, but one gives materially more detail, conditions or exceptions.
- ONE_SIDED: only one document addresses it (one fact list is empty).
Every fact_id must appear in exactly one group. A group may hold several facts from one side.

Return JSON only:
{"pairs": [{"attribute": "<generic name>", "a_fact_ids": [], "b_fact_ids": [],
            "relation": "SAME|DIFFERENT|PARTIAL|ONE_SIDED",
            "explanation": "<one sentence: what is the same or different>"}]}
"""


ABSENCE_CHECK = """One policy document states the following:
  attribute: "<<attribute>>"
  value: "<<value>>"

Below are the most relevant passages from ANOTHER policy document. Does the other document
address this same attribute, even partially or with a different value?

Passages (each starts with its chunk ID):
<<passages>>

Return JSON only:
{"addressed": true, "child_ids": ["<ids of passages that address it>"],
 "value": "<what the other document says, empty if not addressed>",
 "quote": "<exact short quote from that passage, empty if not addressed>",
 "relation": "SAME|DIFFERENT|PARTIAL|ONE_SIDED",
 "explanation": "<one sentence>"}
Use "addressed": false and "relation": "ONE_SIDED" if none of the passages address it.
"""
