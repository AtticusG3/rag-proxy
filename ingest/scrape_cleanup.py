"""Light boilerplate removal before token-chunking scraped text."""

from __future__ import annotations

import re

# Short lines matching common nav/cookie/consent fluff (not body copy).
_BOILERPLATE_LINE = re.compile(
    r"(?i)(cookie(s)?|subscribe|newsletter|sign[\s-]?up|accept all|"
    r"privacy policy|terms of (service|use)|all rights reserved|"
    r"skip to (main )?content|enable javascript|we use cookies|"
    r"manage preferences|gdpr|do not sell)"
)
_WS_RE = re.compile(r"\n{3,}")


def strip_scrape_boilerplate(text: str) -> str:
    """Drop obvious scrape/nav/consent lines before token chunking."""
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if len(stripped) < 200 and _BOILERPLATE_LINE.search(stripped):
            continue
        kept.append(line)
    return _WS_RE.sub("\n\n", "\n".join(kept)).strip()
