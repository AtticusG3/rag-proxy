"""Token counting for context budget when ENABLE_TOKENIZER_ESTIMATE=true."""

from __future__ import annotations

import logging

from rag_proxy.config import settings

log = logging.getLogger("rag-proxy")

_ENCODER = None
_ENCODING_NAME = "cl100k_base"


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        import tiktoken

        _ENCODER = tiktoken.get_encoding(_ENCODING_NAME)
    return _ENCODER


def uses_tokenizer() -> bool:
    return settings.enable_tokenizer_estimate


def count_tokens(text: str) -> int:
    """Count tokens; falls back to chars/4 when tokenizer estimate is disabled."""
    if not text:
        return 0
    if not uses_tokenizer():
        return max(1, len(text) // 4)
    try:
        return len(_get_encoder().encode(text))
    except Exception:
        log.warning("tokenizer encode failed; using char/4 fallback", exc_info=True)
        return max(1, len(text) // 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens (char slice when estimate disabled)."""
    if max_tokens <= 0 or not text:
        return ""
    if not uses_tokenizer():
        return text[: max_tokens * 4]
    try:
        enc = _get_encoder()
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])
    except Exception:
        log.warning("tokenizer truncate failed; using char fallback", exc_info=True)
        return text[: max_tokens * 4]
