"""LLM helpers: chat (Google Gemini or OpenAI, chosen in config.yaml -> models.provider) with
retries, throttling and JSON parsing; embeddings with a LOCAL sentence-transformers model only
(config.yaml -> models.embeddings, default all-MiniLM-L6-v2) plus a disk cache.

Every LLM call goes through `call_json`, which returns a plain dict:
    {"data": parsed JSON, "raw": raw text, "usage": {...}, "latency_s": float, "model": str}
so callers can save the raw response next to the parsed one.
"""

import json
import re
import threading
import time
import warnings

# LLM client libraries that deserialize LangChain messages with load() trigger a
# LangChainPendingDeprecationWarning about `allowed_objects` on EVERY call. It is harmless but
# floods the console and hides the progress lines, so only that specific message is silenced.
warnings.filterwarnings("ignore", message=r".*allowed_objects.*")

import numpy as np
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.common.utils import PROJECT_ROOT, append_jsonl, read_jsonl, short_hash

load_dotenv(PROJECT_ROOT / ".env")

_last_call = [0.0]          # time of the previous API call (for simple throttling)
_throttle_lock = threading.Lock()
_settings = {"min_interval_s": 4.0, "max_attempts": 6, "timeout_s": 120, "provider": "google",
             "quota_error_patterns": []}


class QuotaExhausted(Exception):
    """Raised when the daily quota is used up: retrying is pointless, the run should stop and resume later."""


def configure(llm_cfg, provider=None):
    """Apply the `llm` block from config.yaml (throttle, retries, timeout) and the provider name."""
    _settings.update({k: v for k, v in (llm_cfg or {}).items() if k in _settings})
    if provider:
        _settings["provider"] = provider


def model_settings(config):
    """Return {"provider", "generator", "judge", "embeddings", "temperature"} for the chosen provider,
    and apply the provider + `llm` settings to this module."""
    m = config["models"]
    provider = m.get("provider", "google")
    configure(config.get("llm"), provider)
    out = {"provider": provider, "temperature": m.get("temperature", 0), "seed": m.get("seed"), **m[provider]}
    if m.get("embeddings"):                       # provider-independent embedding model (e.g. local MiniLM)
        out["embeddings"] = m["embeddings"]
    return out


def get_chat(model, temperature=0, json_mode=True, seed=None):
    """Create a chat model for the configured provider. json_mode=True asks for pure JSON output.
    seed (OpenAI only) makes repeated runs as reproducible as the provider allows."""
    if _settings["provider"] == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {"model": model, "max_retries": 1, "timeout": _settings["timeout_s"], "stream_usage": True}
        if seed is not None:
            kwargs["seed"] = seed
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode:                               # prompts must mention "JSON" - ours all do
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs = {"model": model, "temperature": temperature, "max_retries": 1, "timeout": _settings["timeout_s"]}
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    return ChatGoogleGenerativeAI(**kwargs)


_local_models = {}


def is_local_embedding(model):
    """True for a local sentence-transformers model (a hub name like 'sentence-transformers/...' or a folder path)."""
    from pathlib import Path
    return model.startswith("sentence-transformers/") or Path(model).exists()


def get_embedder(model):
    """Load the local sentence-transformers embedding model (e.g. all-MiniLM-L6-v2, 384 dims).

    Only local models are allowed - no embedding API is ever called. A hub name
    ('sentence-transformers/...') is downloaded once; a folder path works offline.
    """
    if not is_local_embedding(model):
        raise ValueError(f"Embedding model '{model}' is not a local sentence-transformers model. "
                         "Set models.embeddings to e.g. sentence-transformers/all-MiniLM-L6-v2 or a local folder path.")
    if model not in _local_models:
        from sentence_transformers import SentenceTransformer
        _local_models[model] = SentenceTransformer(model)
    return {"local": True, "name": model, "st": _local_models[model]}


# ------------------------------------------------------------------ retry rules

def _is_retryable(exc):
    """Retry on rate limits, server errors, timeouts and bad JSON; never on auth / bad request / quota."""
    msg = str(exc)
    if isinstance(exc, QuotaExhausted):
        return False
    if isinstance(exc, ValueError) and msg.startswith("Bad JSON"):
        return True
    if any(s in msg for s in ("400", "401", "403", "404", "INVALID_ARGUMENT", "PERMISSION_DENIED", "NOT_FOUND",
                              "invalid_api_key", "model_not_found")):
        return False
    return True


def _check_quota(exc):
    """Turn a daily-quota / no-credit error into QuotaExhausted so the whole run stops cleanly."""
    msg = str(exc)
    gemini_daily = ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and re.search(r"per ?day", msg, re.IGNORECASE)
    openai_no_credit = "insufficient_quota" in msg
    custom = any(re.search(p, msg, re.IGNORECASE) for p in _settings["quota_error_patterns"])  # e.g. in-house LLM
    if gemini_daily or openai_no_credit or custom:
        raise QuotaExhausted(msg[:500]) from exc


def _throttle():
    """Wait so that calls start at least `min_interval_s` apart (thread-safe; keeps under rate limits)."""
    with _throttle_lock:
        wait = _settings["min_interval_s"] - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


def _with_retry(fn):
    """Wrap fn with exponential-backoff retries using the configured attempt count."""
    return retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(_settings["max_attempts"]),
        wait=wait_exponential(multiplier=2, min=4, max=90),
        reraise=True,
    )(fn)


# ------------------------------------------------------------------ chat

def message_text(msg):
    """Plain text of a LangChain AI message (handles content returned as a list of blocks)."""
    text = getattr(msg, "text", None)
    if isinstance(text, str):                 # newer LangChain: property (a str subclass)
        return str(text)
    if callable(text):                        # older LangChain: method
        return text()
    content = msg.content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def parse_json(text):
    """Parse JSON from model text, tolerating ```json fences or extra prose around it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise


def call_json(llm, prompt, required_keys=()):
    """Call the model, parse its JSON and check required top-level keys. Retries on failure.

    Raises QuotaExhausted for daily-quota errors, or the last error after all retries.
    """
    def once():
        _throttle()
        start = time.time()
        try:
            msg = llm.invoke(prompt)
        except Exception as exc:
            _check_quota(exc)
            raise
        latency = time.time() - start
        raw = message_text(msg)
        try:
            data = parse_json(raw)
        except Exception:
            raise ValueError(f"Bad JSON from model: {raw[:3000]}")
        missing = [k for k in required_keys if not isinstance(data, dict) or k not in data]
        if missing:
            raise ValueError(f"Bad JSON from model (missing keys {missing}): {raw[:3000]}")
        usage = dict(getattr(msg, "usage_metadata", None) or {})
        return {"data": data, "raw": raw, "usage": usage, "latency_s": round(latency, 3),
                "model": getattr(llm, "model", None) or getattr(llm, "model_name", "")}

    return _with_retry(once)()


def call_json_stream(llm, prompt, required_keys=()):
    """Like call_json, but streams the response to measure time-to-first-token (TTFT).

    Returns the call_json dict plus "ttft_s" (seconds until the first content token).
    """
    def once():
        _throttle()
        start, first, msg = time.time(), None, None
        try:
            for chunk in llm.stream(prompt):
                if first is None and message_text(chunk):
                    first = time.time()
                msg = chunk if msg is None else msg + chunk
        except Exception as exc:
            _check_quota(exc)
            raise
        latency = time.time() - start
        raw = message_text(msg) if msg is not None else ""
        try:
            data = parse_json(raw)
        except Exception:
            raise ValueError(f"Bad JSON from model: {raw[:3000]}")
        missing = [k for k in required_keys if not isinstance(data, dict) or k not in data]
        if missing:
            raise ValueError(f"Bad JSON from model (missing keys {missing}): {raw[:3000]}")
        usage = dict(getattr(msg, "usage_metadata", None) or {})
        return {"data": data, "raw": raw, "usage": usage, "latency_s": round(latency, 3),
                "ttft_s": round((first or time.time()) - start, 3),
                "model": getattr(llm, "model", None) or getattr(llm, "model_name", "")}

    return _with_retry(once)()


# ------------------------------------------------------------------ embeddings

def embed_texts(embedder, texts, cache_path):
    """Embed texts with the local model, reusing vectors cached on disk (keyed by model + text hash).
    Vectors are L2-normalised. Returns a numpy array (one row per text)."""
    model = embedder["name"]
    cache = {r["key"]: r["vector"] for r in read_jsonl(cache_path)}
    keys = [f"{model}:{short_hash(t, 16)}" for t in texts]
    todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in cache]

    for i in range(0, len(todo), 50):                       # small batches
        batch = todo[i:i + 50]
        vectors = embedder["st"].encode([t for _, t in batch], normalize_embeddings=True).tolist()
        for (k, _), v in zip(batch, vectors):
            cache[k] = v
            append_jsonl(cache_path, {"key": k, "vector": v})
    return np.array([cache[k] for k in keys], dtype=float)


def cosine_matrix(a, b):
    """Cosine similarity between every row of a and every row of b."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T
