"""R2 typo / noise perturbations. Deterministic: the same seed + question always gives the same variant."""

import random
import string


def rng_for(seed, *parts):
    """Random generator seeded by the suite seed plus identifiers (e.g. qid + perturbation type)."""
    return random.Random(":".join([str(seed), *map(str, parts)]))


def char_typos(text, rate, rng):
    """Edit about `rate` of the letters: swap with the next letter, delete, duplicate or substitute."""
    chars = list(text)
    letters = [i for i, c in enumerate(chars) if c.isalpha()]
    n = max(1, round(rate * len(letters)))
    for i in sorted(rng.sample(letters, min(n, len(letters))), reverse=True):   # right to left keeps indices valid
        op = rng.choice(["swap", "delete", "duplicate", "substitute"])
        if op == "swap" and i + 1 < len(chars) and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "delete":
            del chars[i]
        elif op == "duplicate":
            chars.insert(i, chars[i])
        else:
            new = rng.choice(string.ascii_lowercase)
            chars[i] = new.upper() if chars[i].isupper() else new
    return "".join(chars)


def case_noise(text, rate, rng):
    """Change the casing of about `rate` of the words (UPPER, lower or Title)."""
    words = text.split(" ")
    for i, w in enumerate(words):
        if w and rng.random() < rate:
            words[i] = rng.choice([w.upper(), w.lower(), w.title()])
    return " ".join(words)


def space_noise(text, rate, rng):
    """Add or remove spaces between about `rate` of the words, plus stray leading/trailing spaces."""
    words, out = text.split(" "), []
    for i, w in enumerate(words):
        out.append(w)
        if i < len(words) - 1:
            r = rng.random()
            out.append("" if r < rate / 2 else ("   " if r < rate else " "))
    return "  " + "".join(out) + "  "


PERTURBATIONS = {"typo_char": char_typos, "typo_case": case_noise, "typo_space": space_noise}
