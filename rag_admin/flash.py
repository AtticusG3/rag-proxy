"""One-shot flash messages via query string on redirect."""

from __future__ import annotations

from urllib.parse import urlencode

from fastapi.responses import RedirectResponse


def flash_redirect(url: str, message: str, *, level: str = "info") -> RedirectResponse:
    query = urlencode({"flash": message, "flash_level": level})
    separator = "&" if "?" in url else "?"
    return RedirectResponse(url=f"{url}{separator}{query}", status_code=303)
