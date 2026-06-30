"""Shared Jinja2 environment for rag-admin templates."""

from __future__ import annotations

import os

from fastapi.templating import Jinja2Templates

from rag_admin.helpers import format_datetime

_BASE = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(_BASE, "templates"))
templates.env.filters["format_dt"] = format_datetime
