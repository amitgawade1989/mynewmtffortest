"""Deterministic NLI classifier (local cross-encoder) used by P5 and P6."""

_model = {}


def load_nli(name):
    """Load the cross-encoder once and reuse it."""
    if name not in _model:
        from sentence_transformers import CrossEncoder

        _model[name] = CrossEncoder(name)
    return _model[name]


def nli_probs(model_name, pairs, batch_size=16):
    """Classify (premise, hypothesis) pairs. Returns [{"entailment": p, "contradiction": p, "neutral": p}]."""
    if not pairs:
        return []
    model = load_nli(model_name)
    id2label = model.model.config.id2label
    scores = model.predict(pairs, apply_softmax=True, batch_size=batch_size, show_progress_bar=False)
    return [{id2label[i].lower(): float(s[i]) for i in range(len(s))} for s in scores]
