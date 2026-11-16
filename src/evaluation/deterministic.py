"""Deterministic metric helpers (no LLM): text normalisation, token F1, numbers, latency, cost, citation IDs."""

import re
import string
from collections import Counter
from statistics import mean, median


def normalize_answer(text):
    """Lowercase, strip punctuation / articles / extra spaces (SQuAD-style normalisation)."""
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction, reference):
    """Token-overlap F1 between two texts (P1 extractive measure)."""
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    common = sum((Counter(p) & Counter(r)).values())
    if not p or not r or not common:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30}


def extract_numbers(text):
    """All numbers in a text, as floats ("37.5", "4x", "1,500", "two" -> 2.0)."""
    text = text or ""
    nums = [float(m.replace(",", "")) for m in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)]
    nums += [float(NUMBER_WORDS[w]) for w in re.findall(r"[a-z]+", text.lower()) if w in NUMBER_WORDS]
    return nums


def numeric_match(prediction, reference, rel_tol=0.01):
    """Share of the reference's numbers that also appear in the prediction (within tolerance).

    Returns None when the reference contains no numbers (metric not applicable).
    """
    ref = sorted(set(extract_numbers(reference)))
    if not ref:
        return None
    pred = extract_numbers(prediction)
    found = sum(any(abs(p - r) <= max(1e-9, rel_tol * abs(r)) for p in pred) for r in ref)
    return found / len(ref)


def split_sentences(text):
    """Simple sentence splitter (keeps decimals like 37.5 together)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", (text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]


def set_prf(predicted, gold):
    """Precision, recall and F1 of a predicted ID set against a gold ID set (None if undefined)."""
    predicted, gold = set(predicted), set(gold)
    p = len(predicted & gold) / len(predicted) if predicted else None
    r = len(predicted & gold) / len(gold) if gold else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return p, r, f


def percentile(values, q):
    """q-th percentile (0-100) with linear interpolation."""
    vals = sorted(values)
    if not vals:
        return None
    pos = (len(vals) - 1) * q / 100
    lo, hi = int(pos), min(int(pos) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def latency_stats(values):
    """mean / p50 / p95 / max of a list of seconds."""
    if not values:
        return {}
    return {"mean": round(mean(values), 3), "p50": round(median(values), 3),
            "p95": round(percentile(values, 95), 3), "max": round(max(values), 3)}


def cost_usd(usage, model, pricing):
    """Cost of one call from token usage and a {model: {input, output}} price table (USD per 1M tokens)."""
    rates = pricing.get(model)
    if not rates or not usage:
        return None
    return usage.get("input_tokens", 0) / 1e6 * rates["input"] + usage.get("output_tokens", 0) / 1e6 * rates["output"]
